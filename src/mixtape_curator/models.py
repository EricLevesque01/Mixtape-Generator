from typing import List, Optional, Dict, Tuple
from pydantic import BaseModel, Field

class RYMData(BaseModel):
    primary_genres: List[str] = Field(default_factory=list)
    subgenres: List[str] = Field(default_factory=list)
    descriptors: List[str] = Field(default_factory=list)

class Track(BaseModel):
    id: str
    title: str
    artist: str
    album: Optional[str] = None
    duration_s: int
    rym_data: RYMData = Field(default_factory=RYMData)
    
    # Sonic features (normalized 0.0 - 1.0)
    energy: float = 0.0
    valence: float = 0.0
    intensity: float = 0.0
    tempo: float = 120.0
    danceability: float = 0.5
    key: int = 0
    mode: int = 1
    key_full: str = "Unknown"
    acousticness: float = 0.0
    instrumentalness: float = 0.0
    speechiness: float = 0.0
    liveness: float = 0.0
    brightness: float = 0.0
    flatness: float = 0.0
    entropy: float = 0.0
    dynamic_range: float = 0.0
    
    # Metadata features
    accessibility: float = 0.6
    familiarity: float = 0.5
    rating: float = 0.0
    recommendability: float = 0.5
    liked: bool = False       # iTunes track-level "Loved" heart
    album_loved: bool = False  # iTunes album-level "Loved" heart

    # Mix curation signals
    mix_appearances: int = 0        # raw # of playlists track appears in
    mix_prominence: float = 0.0     # weighted + normalized (0-1), cap=10
    artist_mix_prominence: float = 0.0  # artist-level aggregate (0-1)
    spotify_affinity: float = 0.0   # Spotify listening signal (0-1)
    
    # System fields
    file_path: Optional[str] = None
    spotify_uri: Optional[str] = None
    enrichment_source: Optional[str] = None
    release_year: Optional[int] = None
    track_number: Optional[int] = None
    total_tracks: Optional[int] = None
    disc_number: Optional[int] = None
    enrichment_gaps: List[str] = Field(default_factory=list)  # Fields that were missing/defaulted
    
    @property
    def normalized_artist(self) -> str:
        return self.artist.lower().strip()

class FeedbackTargets(BaseModel):
    uniformity: float = 0.35  # 0.0 = Eclectic, 1.0 = Uniform. Default 0.35 favors "Segmented/Eclectic".
    energy: float = 0.5
    valence: float = 0.5
    intensity: float = 0.5
    accessibility: float = 0.5
    familiarity: float = 0.5

# ---------------------------------------------------------------------------
# Persona → Music Mapping
# ---------------------------------------------------------------------------

_OCCASION_MAP: Dict[str, Tuple[float, float]] = {
    # (energy, accessibility)   defaults: energy=0.5, accessibility=0.5
    "gym": (0.90, 0.80),
    "workout": (0.90, 0.80),
    "run": (0.85, 0.75),
    "party": (0.80, 0.85),
    "drive": (0.65, 0.70),
    "road trip": (0.65, 0.70),
    "study": (0.30, 0.55),
    "focus": (0.30, 0.50),
    "late night": (0.35, 0.35),
    "sleep": (0.15, 0.40),
    "chill": (0.35, 0.55),
    "dinner": (0.45, 0.65),
    "pregame": (0.75, 0.80),
}

_AESTHETIC_MAP: Dict[str, Tuple[float, float, float]] = {
    # (intensity, valence, genre_strictness)
    "dark room": (0.70, 0.25, 0.50),
    "headphones": (0.70, 0.25, 0.50),
    "rooftop": (0.50, 0.70, 0.30),
    "sunset": (0.50, 0.70, 0.30),
    "basement": (0.85, 0.40, 0.40),
    "basement show": (0.85, 0.40, 0.40),
    "sweaty": (0.85, 0.40, 0.40),
}

_PERSONALITY_DESCRIPTOR_MAP: Dict[str, List[str]] = {
    "chill": ["relaxed", "atmospheric", "gentle"],
    "introspective": ["melancholic", "pensive", "slow"],
    "adventurous": ["eclectic", "experimental", "energetic"],
    "energetic": ["driving", "upbeat", "intense"],
    "creative": ["experimental", "art pop", "avant-garde"],
    "nostalgic": ["warm", "dreamy", "lush"],
    "dark": ["brooding", "moody", "heavy"],
    "happy": ["bright", "joyful", "playful"],
    "chaotic": ["dissonant", "frenetic", "raw"],
    "laid back": ["lo-fi", "hazy", "smooth"],
    "curious": ["complex", "layered", "sophisticated"],
    "intense": ["powerful", "aggressive", "dense"],
    "romantic": ["lush", "warm", "intimate"],
    "mysterious": ["dark", "ethereal", "haunting"],
}

_MOOD_ENERGY_MAP: Dict[str, float] = {
    "happy": 0.70, "joyful": 0.80, "excited": 0.85,
    "anxious": 0.65, "stressed": 0.60, "hyped": 0.90,
    "sad": 0.25, "down": 0.20, "melancholic": 0.30,
    "lonely": 0.25, "numb": 0.15,
    "calm": 0.35, "relaxed": 0.30, "peaceful": 0.25,
    "focused": 0.45, "productive": 0.50, "motivated": 0.65,
    "nostalgic": 0.40, "reflective": 0.35, "introspective": 0.35,
    "angry": 0.80, "frustrated": 0.70,
    "bored": 0.50, "restless": 0.60,
}


class PersonaProfile(BaseModel):
    """Raw answers from the persona interview before mapping to music parameters."""
    occasion: str = ""
    personality_words: List[str] = Field(default_factory=list)
    mood_today: str = ""
    aesthetic_choice: str = ""   # e.g. "dark room", "rooftop at sunset", "basement show"
    era_preference: str = ""     # "nostalgia" | "new" | "both"
    wildcard: str = ""

    # §3A — per-axis confidence (0.0 = no signal, 1.0 = fully populated)
    # Keys: occasion, personality, mood, aesthetic, era, wildcard
    confidence: Dict[str, float] = Field(default_factory=dict)

    def to_user_profile(self) -> "UserProfile":
        """Map persona signals → UserProfile for the generator."""
        targets = FeedbackTargets()
        descriptors: List[str] = []
        genre_strictness = 0.30  # default — fairly open

        # --- Occasion → energy + accessibility ---
        occasion_lower = self.occasion.lower()
        for keyword, (eng, acc) in _OCCASION_MAP.items():
            if keyword in occasion_lower:
                targets.energy = eng
                targets.accessibility = acc
                break
        else:
            targets.energy = 0.50
            targets.accessibility = 0.55

        # --- Mood → energy refinement + valence ---
        mood_lower = self.mood_today.lower()
        for keyword, eng in _MOOD_ENERGY_MAP.items():
            if keyword in mood_lower:
                # Blend occasion energy (60%) with mood energy (40%)
                targets.energy = round(targets.energy * 0.60 + eng * 0.40, 2)
                # Positive moods raise valence
                if keyword in ["happy", "joyful", "excited", "hyped", "peaceful", "calm"]:
                    targets.valence = 0.70
                elif keyword in ["sad", "down", "melancholic", "lonely", "numb"]:
                    targets.valence = 0.20
                elif keyword in ["angry", "frustrated"]:
                    targets.valence = 0.30
                    targets.intensity = 0.80
                else:
                    targets.valence = 0.50
                break
        else:
            targets.valence = 0.50

        # --- Aesthetic → intensity + valence refinement + strictness ---
        aesthetic_lower = self.aesthetic_choice.lower()
        for keyword, (inten, val, strict) in _AESTHETIC_MAP.items():
            if keyword in aesthetic_lower:
                targets.intensity = inten
                # Blend valence
                targets.valence = round(targets.valence * 0.50 + val * 0.50, 2)
                genre_strictness = strict
                break
        else:
            targets.intensity = 0.50

        # --- Personality → descriptors + uniformity ---
        for word in self.personality_words:
            word_lower = word.lower().strip()
            for key, desc_list in _PERSONALITY_DESCRIPTOR_MAP.items():
                if key in word_lower:
                    descriptors.extend(desc_list)

        # Uniformity: eclectic personalities → lower; focused/calm → higher
        eclectic_words = {"adventurous", "chaotic", "creative", "curious"}
        uniform_words = {"focused", "calm", "chill", "relaxed", "peaceful"}
        personality_set = {w.lower() for w in self.personality_words}
        if personality_set & eclectic_words:
            targets.uniformity = 0.20
        elif personality_set & uniform_words:
            targets.uniformity = 0.70
        else:
            targets.uniformity = 0.40  # balanced default

        # --- Era preference → familiarity ---
        era_lower = self.era_preference.lower()
        if "nostalgia" in era_lower or "old" in era_lower or "classic" in era_lower:
            targets.familiarity = 0.75
        elif "new" in era_lower or "discover" in era_lower or "fresh" in era_lower:
            targets.familiarity = 0.25
        else:
            targets.familiarity = 0.50

        # --- Wildcard → extra descriptors ---
        wildcard_lower = self.wildcard.lower()
        for key, desc_list in _PERSONALITY_DESCRIPTOR_MAP.items():
            if key in wildcard_lower:
                descriptors.extend(desc_list)

        # Deduplicate descriptors
        descriptors = list(dict.fromkeys(descriptors))

        # Build a human-readable context note
        context = self.occasion or "personal listening"
        if self.mood_today:
            context += f" / {self.mood_today} mood"

        return UserProfile(
            recipient="you",
            context_notes=context,
            target_descriptors=descriptors,
            genre_strictness=genre_strictness,
            targets=targets,
        )


class UserProfile(BaseModel):
    recipient: str = "self"
    context_notes: Optional[str] = None
    
    must_include_track_ids: List[str] = Field(default_factory=list)
    must_include_artists: List[str] = Field(default_factory=list)
    
    exclude_track_ids: List[str] = Field(default_factory=list)
    exclude_artists: List[str] = Field(default_factory=list)
    exclude_genres: List[str] = Field(default_factory=list)
    exclude_descriptors: List[str] = Field(default_factory=list)
    
    target_genres: List[str] = Field(default_factory=list)
    target_descriptors: List[str] = Field(default_factory=list)
    
    genre_strictness: float = 0.3 # Default blend
    
    targets: FeedbackTargets = Field(default_factory=FeedbackTargets)
    
    # Duration override — set by interview if user states a target (e.g. "120 minutes")
    # When None, falls back to config.duration_target_s
    duration_target_s: Optional[int] = None

class PlaylistScores(BaseModel):
    fit: float = 0.0
    flow: float = 0.0
    variety: float = 0.0
    accessibility: float = 0.0
    quality: float = 0.0
    total: float = 0.0

class Playlist(BaseModel):
    id: str
    track_ids: List[str] = Field(default_factory=list)
    total_duration_s: int = 0
    scores: PlaylistScores = Field(default_factory=PlaylistScores)
    violations: List[str] = Field(default_factory=list)
    generation_notes: Optional[str] = None
    track_notes: Dict[str, str] = Field(default_factory=dict)  # track_id -> curator reasoning
    # §3E — LLM rationale (required before refinement exits)
    rationale: Optional[str] = None
    # §3D — which segments were used (populated by arc optimizer)
    segments_used: List[str] = Field(default_factory=list)

# ---------------------------------------------------------------------------
# V5 Architecture Contract — §3B, §3C, §3D, §4
# ---------------------------------------------------------------------------

class Segment(BaseModel):
    """
    §4 — A narrative chapter with explicit selection targets.
    Produced by the Blueprint LLM call; consumed by the Trellis and Arc Optimizer.
    """
    segment_id: str
    theme: str

    # Exactly one of these must be set (§3B locked rule)
    target_duration_s: Optional[int] = None
    target_track_count: Optional[int] = None

    # Optional guidance for the trellis (algorithm-readable; no creative inference)
    audio_targets: Optional[Dict[str, float]] = None    # e.g. {"energy": 0.7, "valence": 0.5}
    semantic_targets: Optional[Dict[str, List[str]]] = None  # e.g. {"genres": [...], "descriptors": [...]}
    uniformity_bias: Optional[float] = None            # 0.0 eclectic → 1.0 uniform
    notes: Optional[str] = None


class Blueprint(BaseModel):
    """
    §3B — Segment pool output by the LLM Creative Director.
    Contains 7–8 Segments describing intent, not exact tracks.
    """
    blueprint_id: str
    persona_state_id: str           # links back to the PersonaProfile that generated this
    narrative_arc: str              # Plain-text description of the overall arc
    segments: List[Segment] = Field(default_factory=list)   # 7–8 entries
    # Optional library vocab snapshot used at generation time (for traceability)
    vocab_summary_ref: Optional[str] = None


class SegmentTrellis(BaseModel):
    """
    §3C — Deterministic candidate pools produced by the algorithm.
    Each segment_id maps to a list of track_ids (target ≥ 20 per segment).
    """
    trellis_id: str
    blueprint_id: str
    # segment_id -> ordered list of candidate track_ids (sorted by relevance)
    segment_pools: Dict[str, List[str]] = Field(default_factory=dict)
    # Diagnostic: segments that could not reach ≥20 candidates
    undersized_segments: List[str] = Field(default_factory=list)
    # Reason codes for any hard-constraint failures
    diagnostic_codes: Dict[str, str] = Field(default_factory=dict)


class ArcDraft(BaseModel):
    """
    §3D — A fully assembled arc candidate produced by the arc optimizer.
    Two of these (A and B) are produced per generation run.
    §7 guarantees A and B differ by ≥1 segment and Jaccard(segment sets) ≤ 0.8.
    """
    draft_id: str
    label: str                          # "A" or "B"
    selected_segment_ids: List[str] = Field(default_factory=list)   # 3–5 segments chosen
    tracks_per_segment: Dict[str, List[str]] = Field(default_factory=dict)  # seg_id -> [track_id]
    global_order: List[str] = Field(default_factory=list)           # final flat track_id order
    scores: PlaylistScores = Field(default_factory=PlaylistScores)
    total_duration_s: int = 0
    boundary_flow_score: float = 0.0    # §6 — cross-segment transition quality
    duration_error_s: int = 0           # |total_duration_s - duration_target_s|
    # §10 — failure diagnostics (populated if arc could not be fully satisfied)
    diagnostic_codes: List[str] = Field(default_factory=list)


class EvalResult(BaseModel):
    case_id: str
    model: str
    temperature: float
    seed: Optional[int]
    question_count: int
    repair_iterations: int
    hard_constraint_pass: bool
    final_score: float
    ab_difference: float
    transcript: List[str] = Field(default_factory=list)
