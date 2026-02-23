"""
FastAPI backend for Mixtape Curator Web UI.
Provides WebSocket chat, REST API for playlist CRUD, library search, and exports.
"""

import logging
import json
import asyncio
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from mixtape_curator.library import library
from mixtape_curator.config import config
from mixtape_curator.models import UserProfile, Playlist, PlaylistScores
from mixtape_curator.generator import generator
from mixtape_curator.agent import ReActAgent
from mixtape_curator.ab_test import ab_tester
from mixtape_curator.scoring import scorer
from mixtape_curator.exporter import exporter
from mixtape_curator.spotify_export import spotify_exporter
from mixtape_curator.llm.providers.local import MockLLM

logger = logging.getLogger("mixtape_curator.web")

# ── App Setup ──────────────────────────────────────────────────────────

app = FastAPI(title="Mixtape Curator", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load library at startup
@app.on_event("startup")
async def startup():
    library.load()
    logger.info(f"Library loaded: {len(library.df)} tracks")

# ── Session Store ──────────────────────────────────────────────────────

sessions: dict = {}  # session_id -> { profile, playlist_a, playlist_b, chat_history, interviewer }


# ── Pydantic Models ────────────────────────────────────────────────────

class TrackAction(BaseModel):
    action: str  # "add", "remove", "swap", "reorder"
    track_id: Optional[str] = None
    new_track_id: Optional[str] = None
    new_order: Optional[list] = None
    playlist_label: str = "A"  # "A" or "B"

class SearchQuery(BaseModel):
    query: str
    limit: int = 10

class LabelRequest(BaseModel):
    prompt: Optional[str] = None
    title: str = "My Mixtape"
    session_id: str = ""
    playlist_label: str = "A"


# ── WebSocket: AI Chat ─────────────────────────────────────────────────

@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    # Lazy import to avoid circular
    from mixtape_curator.interview_agent import InterviewAgent
    
    # Initialize session
    interviewer = InterviewAgent()
    sessions[session_id] = {
        "interviewer": interviewer,
        "profile": None,
        "playlist_a": None,
        "playlist_b": None,
        "chat_history": [],
    }
    
    # Send greeting
    greeting = interviewer.start()
    await websocket.send_json({
        "type": "message",
        "role": "assistant",
        "content": greeting,
    })
    
    try:
        while True:
            data = await websocket.receive_json()
            user_msg = data.get("message", "")
            
            sessions[session_id]["chat_history"].append({
                "role": "user", "content": user_msg
            })
            
            # Process through interview agent
            def thought_callback(thought: str):
                # We'll batch thoughts and send after
                pass
            
            response, done = interviewer.process_input(user_msg, user_callback=thought_callback)
            
            sessions[session_id]["chat_history"].append({
                "role": "assistant", "content": response
            })
            
            await websocket.send_json({
                "type": "message",
                "role": "assistant",
                "content": response,
            })
            
            if done:
                # Interview complete — generate playlists
                await websocket.send_json({
                    "type": "status",
                    "content": "Generating your mixtape..."
                })
                
                profile = interviewer.profile
                sessions[session_id]["profile"] = profile
                
                # Run generation in thread pool to not block
                playlist_a, playlist_b = await asyncio.get_event_loop().run_in_executor(
                    None, _generate_playlists, profile
                )
                
                sessions[session_id]["playlist_a"] = playlist_a
                sessions[session_id]["playlist_b"] = playlist_b
                
                # Send playlists
                await websocket.send_json({
                    "type": "playlists",
                    "playlist_a": _serialize_playlist(playlist_a),
                    "playlist_b": _serialize_playlist(playlist_b),
                })
                
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {session_id}")


def _generate_playlists(profile: UserProfile):
    """Synchronous playlist generation (runs in thread pool)."""
    use_incremental = True
    if profile.targets.uniformity < 0.6 and len(profile.target_genres) > 1:
        use_incremental = False
    
    draft = generator.create_draft(profile, incremental=use_incremental)
    sequenced = generator.optimize_flow(draft, profile)
    
    llm = MockLLM()
    agent = ReActAgent(llm=llm)
    playlist_a = agent.repair_playlist(sequenced, profile)
    playlist_b = ab_tester.generate_b_side(playlist_a, profile)
    
    return playlist_a, playlist_b


def _serialize_playlist(playlist: Playlist) -> dict:
    """Convert playlist to JSON-safe dict with track details."""
    tracks = []
    for tid in playlist.track_ids:
        t = library.get_track(tid)
        if t:
            tracks.append({
                "id": t.id,
                "title": t.title,
                "artist": t.artist,
                "album": getattr(t, 'album', ''),
                "duration_s": t.duration_s,
                "energy": t.energy,
                "valence": t.valence,
                "genres": t.rym_data.primary_genres[:2] if t.rym_data else [],
                "subgenres": t.rym_data.subgenres[:2] if t.rym_data else [],
                "descriptors": t.rym_data.descriptors[:3] if t.rym_data else [],
                "note": playlist.track_notes.get(tid, ""),
                "spotify_uri": t.spotify_uri,
            })
    
    return {
        "id": playlist.id,
        "tracks": tracks,
        "total_duration_s": playlist.total_duration_s,
        "scores": {
            "fit": playlist.scores.fit,
            "flow": playlist.scores.flow,
            "variety": playlist.scores.variety,
            "accessibility": playlist.scores.accessibility,
            "quality": playlist.scores.quality,
            "total": playlist.scores.total,
        },
        "violations": playlist.violations,
    }


# ── REST: Library Search ───────────────────────────────────────────────

@app.get("/api/library/search")
async def search_library(q: str, limit: int = 10):
    """Search library by genre, descriptor, artist, or title."""
    results = []
    
    # Use the inverted index
    tag_ids = library.search_by_tags(genres=[q], descriptors=[q], limit=limit)
    
    # Also try artist/title match
    artist = library.search_artist(q)
    if artist:
        artist_tracks = library.df[library.df['artist'] == artist]['id'].head(limit).tolist()
        tag_ids = list(dict.fromkeys(artist_tracks + tag_ids))[:limit]
    
    for tid in tag_ids[:limit]:
        t = library.get_track(tid)
        if t:
            results.append({
                "id": t.id,
                "title": t.title,
                "artist": t.artist,
                "album": getattr(t, 'album', ''),
                "duration_s": t.duration_s,
                "genres": t.rym_data.primary_genres[:2] if t.rym_data else [],
                "descriptors": t.rym_data.descriptors[:3] if t.rym_data else [],
            })
    
    return {"results": results, "total": len(results)}


# ── REST: Playlist CRUD ───────────────────────────────────────────────

@app.post("/api/playlist/{session_id}/tracks")
async def modify_playlist(session_id: str, action: TrackAction):
    """Add, remove, swap, or reorder tracks in a playlist."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    
    playlist = session[f"playlist_{action.playlist_label.lower()}"]
    if not playlist:
        raise HTTPException(404, "Playlist not found")
    
    if action.action == "add" and action.track_id:
        if action.track_id not in playlist.track_ids:
            t = library.get_track(action.track_id)
            if t:
                playlist.track_ids.append(action.track_id)
                playlist.total_duration_s += t.duration_s
                playlist.track_notes[action.track_id] = "Added manually by user."
    
    elif action.action == "remove" and action.track_id:
        if action.track_id in playlist.track_ids:
            t = library.get_track(action.track_id)
            playlist.track_ids.remove(action.track_id)
            if t:
                playlist.total_duration_s -= t.duration_s
            playlist.track_notes.pop(action.track_id, None)
    
    elif action.action == "swap" and action.track_id and action.new_track_id:
        if action.track_id in playlist.track_ids:
            idx = playlist.track_ids.index(action.track_id)
            old_t = library.get_track(action.track_id)
            new_t = library.get_track(action.new_track_id)
            if new_t:
                playlist.track_ids[idx] = action.new_track_id
                if old_t:
                    playlist.total_duration_s -= old_t.duration_s
                playlist.total_duration_s += new_t.duration_s
                playlist.track_notes.pop(action.track_id, None)
                playlist.track_notes[action.new_track_id] = "Swapped in by user."
    
    elif action.action == "reorder" and action.new_order:
        # Validate all IDs exist
        if set(action.new_order) == set(playlist.track_ids):
            playlist.track_ids = action.new_order
    
    # Rescore
    profile = session.get("profile") or UserProfile()
    _rescore_playlist(playlist, profile)
    
    return _serialize_playlist(playlist)


def _rescore_playlist(playlist: Playlist, profile: UserProfile):
    """Recalculate all scores for a playlist."""
    try:
        playlist.scores = scorer.score_playlist(playlist, profile)
    except Exception as e:
        logger.warning(f"Rescoring failed (non-fatal): {e}")
        # Manually update total_duration_s at minimum
        tracks = [library.get_track(tid) for tid in playlist.track_ids]
        tracks = [t for t in tracks if t]
        playlist.total_duration_s = sum(t.duration_s for t in tracks)


@app.get("/api/playlist/{session_id}")
async def get_playlists(session_id: str):
    """Get both playlists for a session."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    
    result = {}
    if session.get("playlist_a"):
        result["playlist_a"] = _serialize_playlist(session["playlist_a"])
    if session.get("playlist_b"):
        result["playlist_b"] = _serialize_playlist(session["playlist_b"])
    
    return result


# ── REST: Export ───────────────────────────────────────────────────────

@app.post("/api/export/{session_id}/spotify")
async def export_spotify(session_id: str, playlist_label: str = "A"):
    """Export playlist to Spotify."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    
    playlist = session.get(f"playlist_{playlist_label.lower()}")
    if not playlist:
        raise HTTPException(404, "Playlist not found")
    
    profile = session.get("profile") or UserProfile()
    name = f"Mixtape for {profile.recipient or 'Me'}"
    
    result = await asyncio.get_event_loop().run_in_executor(
        None, spotify_exporter.export_playlist, playlist, name
    )
    
    return {"result": result}


@app.post("/api/export/{session_id}/files")
async def export_files(session_id: str, playlist_label: str = "A"):
    """Export playlist to TXT + M3U8."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    
    playlist = session.get(f"playlist_{playlist_label.lower()}")
    if not playlist:
        raise HTTPException(404, "Playlist not found")
    
    profile = session.get("profile") or UserProfile()
    base = f"mixtape_{profile.recipient or 'custom'}_{playlist_label}"
    
    result = exporter.export_playlist(playlist, base)
    return {"result": result, "base": base}


# ── REST: CD Label ─────────────────────────────────────────────────────

@app.post("/api/label/generate")
async def generate_label(request: LabelRequest):
    """Generate a CD label PDF with optional AI-generated artwork."""
    from mixtape_curator.label_designer import create_cd_label
    
    session = sessions.get(request.session_id) if request.session_id else None
    playlist = None
    if session:
        playlist = session.get(f"playlist_{request.playlist_label.lower()}")
    
    pdf_path = await asyncio.get_event_loop().run_in_executor(
        None,
        create_cd_label,
        request.title,
        request.prompt,
        playlist,
        None,  # image_path (for uploaded images)
    )
    
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"{request.title.replace(' ', '_')}_label.pdf"
    )


@app.post("/api/label/upload")
async def upload_label_image(
    file: UploadFile = File(...),
    title: str = Form("My Mixtape"),
    session_id: str = Form(""),
    playlist_label: str = Form("A"),
):
    """Upload an image for the CD label."""
    from mixtape_curator.label_designer import create_cd_label
    
    # Save uploaded file
    upload_dir = Path("exports/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    img_path = upload_dir / f"{uuid.uuid4().hex}_{file.filename}"
    
    with open(img_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    session = sessions.get(session_id) if session_id else None
    playlist = None
    if session:
        playlist = session.get(f"playlist_{playlist_label.lower()}")
    
    pdf_path = await asyncio.get_event_loop().run_in_executor(
        None,
        create_cd_label,
        title,
        None,  # no AI prompt
        playlist,
        str(img_path),
    )
    
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"{title.replace(' ', '_')}_label.pdf"
    )


# ── New Session ────────────────────────────────────────────────────────

@app.post("/api/session/new")
async def create_session():
    """Create a new session and return its ID."""
    sid = uuid.uuid4().hex[:12]
    sessions[sid] = {
        "interviewer": None,
        "profile": None,
        "playlist_a": None,
        "playlist_b": None,
        "chat_history": [],
    }
    return {"session_id": sid}
