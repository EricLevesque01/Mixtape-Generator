import { useState, useEffect, useRef, useCallback } from 'react'
import {
  DndContext,
  closestCenter,
  PointerSensor,
  useSensor,
  useSensors,
} from '@dnd-kit/core'
import {
  SortableContext,
  verticalListSortingStrategy,
  useSortable,
  arrayMove,
} from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import './App.css'

const API = 'http://localhost:8000'

/* ── Sortable Track Row ──────────────────────────────────── */
function SortableTrack({ track, index, onRemove, formatDuration }) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: track.id })

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
    zIndex: isDragging ? 10 : 'auto',
  }

  return (
    <div ref={setNodeRef} style={style} className={`track-row ${isDragging ? 'dragging' : ''}`}>
      <div className="drag-handle" {...attributes} {...listeners}>⠿</div>
      <div className="track-num">{index + 1}</div>
      <div className="track-info">
        <div className="track-title">{track.title}</div>
        <div className="track-artist">{track.artist}</div>
        {track.note && <div className="track-note">{track.note}</div>}
      </div>
      <div className="track-tags">
        {track.genres.map(g => <span key={g} className="tag tag-genre">{g}</span>)}
        {track.descriptors.slice(0, 2).map(d => <span key={d} className="tag tag-desc">{d}</span>)}
      </div>
      <div className="track-duration">{formatDuration(track.duration_s)}</div>
      <button className="btn-icon btn-remove" onClick={() => onRemove(track.id)} title="Remove">✕</button>
    </div>
  )
}

/* ── Main App ────────────────────────────────────────────── */
function App() {
  const [sessionId, setSessionId] = useState(null)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [playlists, setPlaylists] = useState({ a: null, b: null })
  const [activePlaylist, setActivePlaylist] = useState('a')
  const [generating, setGenerating] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [showSearch, setShowSearch] = useState(false)
  const [showLabel, setShowLabel] = useState(false)
  const [labelTitle, setLabelTitle] = useState('My Mixtape')
  const [labelPrompt, setLabelPrompt] = useState('')
  const wsRef = useRef(null)
  const chatEndRef = useRef(null)

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 5 } })
  )

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Create session + WebSocket on mount
  useEffect(() => {
    fetch(`${API}/api/session/new`, { method: 'POST' })
      .then(r => r.json())
      .then(data => {
        setSessionId(data.session_id)
        const ws = new WebSocket(`ws://localhost:8000/ws/chat/${data.session_id}`)

        ws.onmessage = (event) => {
          const msg = JSON.parse(event.data)
          if (msg.type === 'message') {
            setMessages(prev => [...prev, { role: msg.role, content: msg.content }])
          } else if (msg.type === 'status') {
            setMessages(prev => [...prev, { role: 'system', content: msg.content }])
            setGenerating(true)
          } else if (msg.type === 'playlists') {
            setPlaylists({ a: msg.playlist_a, b: msg.playlist_b })
            setGenerating(false)
            setMessages(prev => [...prev, {
              role: 'system',
              content: '✨ Your mixtapes are ready! Check them out on the right →'
            }])
          }
        }

        wsRef.current = ws
      })

    return () => wsRef.current?.close()
  }, [])

  const sendMessage = useCallback(() => {
    if (!input.trim() || !wsRef.current) return
    setMessages(prev => [...prev, { role: 'user', content: input }])
    wsRef.current.send(JSON.stringify({ message: input }))
    setInput('')
  }, [input])

  // Track actions
  const removeTrack = async (trackId) => {
    try {
      const res = await fetch(`${API}/api/playlist/${sessionId}/tracks`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'remove', track_id: trackId, playlist_label: activePlaylist.toUpperCase() })
      })
      if (res.ok) {
        const updated = await res.json()
        setPlaylists(prev => ({ ...prev, [activePlaylist]: updated }))
      }
    } catch (err) {
      console.error('Remove failed:', err)
    }
  }

  const addTrack = async (trackId) => {
    try {
      const res = await fetch(`${API}/api/playlist/${sessionId}/tracks`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'add', track_id: trackId, playlist_label: activePlaylist.toUpperCase() })
      })
      if (res.ok) {
        const updated = await res.json()
        setPlaylists(prev => ({ ...prev, [activePlaylist]: updated }))
      }
      setShowSearch(false)
    } catch (err) {
      console.error('Add failed:', err)
    }
  }

  const reorderTracks = async (newOrder) => {
    try {
      const res = await fetch(`${API}/api/playlist/${sessionId}/tracks`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'reorder', new_order: newOrder, playlist_label: activePlaylist.toUpperCase() })
      })
      if (res.ok) {
        const updated = await res.json()
        setPlaylists(prev => ({ ...prev, [activePlaylist]: updated }))
      }
    } catch (err) {
      console.error('Reorder failed:', err)
    }
  }

  const handleDragEnd = (event) => {
    const { active, over } = event
    if (!over || active.id === over.id) return

    const currentPlaylist = playlists[activePlaylist]
    if (!currentPlaylist) return

    const oldIndex = currentPlaylist.tracks.findIndex(t => t.id === active.id)
    const newIndex = currentPlaylist.tracks.findIndex(t => t.id === over.id)

    // Optimistic update — move tracks locally first for instant feedback
    const newTracks = arrayMove(currentPlaylist.tracks, oldIndex, newIndex)
    const updatedPlaylist = { ...currentPlaylist, tracks: newTracks }
    setPlaylists(prev => ({ ...prev, [activePlaylist]: updatedPlaylist }))

    // Then sync with backend for rescoring
    const newOrder = newTracks.map(t => t.id)
    reorderTracks(newOrder)
  }

  const searchLibrary = async () => {
    if (!searchQuery.trim()) return
    const res = await fetch(`${API}/api/library/search?q=${encodeURIComponent(searchQuery)}&limit=10`)
    const data = await res.json()
    setSearchResults(data.results)
  }

  const exportSpotify = async () => {
    const res = await fetch(`${API}/api/export/${sessionId}/spotify?playlist_label=${activePlaylist.toUpperCase()}`, { method: 'POST' })
    const data = await res.json()
    alert(data.result)
  }

  const downloadLabel = async () => {
    const res = await fetch(`${API}/api/label/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title: labelTitle,
        prompt: labelPrompt || null,
        session_id: sessionId,
        playlist_label: activePlaylist.toUpperCase(),
      })
    })
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${labelTitle.replace(/\s+/g, '_')}_label.pdf`
    a.click()
    URL.revokeObjectURL(url)
  }

  const exportFiles = async () => {
    const res = await fetch(`${API}/api/export/${sessionId}/files?playlist_label=${activePlaylist.toUpperCase()}`, { method: 'POST' })
    const data = await res.json()
    alert(data.result)
  }

  const currentPlaylist = playlists[activePlaylist]

  const formatDuration = (s) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
  const formatTotalDuration = (s) => `${Math.floor(s / 60)}m ${s % 60}s`

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="logo">🎵 Mixtape Curator</div>
        <div className="header-actions">
          {currentPlaylist && (
            <>
              <button className="btn btn-outline" onClick={() => setShowSearch(true)}>+ Add Track</button>
              <button className="btn btn-outline" onClick={() => setShowLabel(true)}>💿 CD Label</button>
              <button className="btn btn-outline" onClick={exportFiles}>📁 Export Files</button>
              <button className="btn btn-spotify" onClick={exportSpotify}>
                <span className="spotify-icon">●</span> Export to Spotify
              </button>
            </>
          )}
        </div>
      </header>

      <main className="main">
        {/* Chat Panel */}
        <section className="chat-panel">
          <div className="panel-header">
            <h2>🎤 AI Curator</h2>
          </div>
          <div className="chat-messages">
            {messages.map((msg, i) => (
              <div key={i} className={`message message-${msg.role}`}>
                <div className="message-bubble">
                  {msg.role === 'system' && <span className="system-badge">System</span>}
                  {msg.content}
                </div>
              </div>
            ))}
            {generating && (
              <div className="message message-system">
                <div className="message-bubble generating">
                  <span className="dot-pulse"></span> Generating your mixtape...
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>
          <div className="chat-input">
            <input
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && sendMessage()}
              placeholder="Describe your perfect mixtape..."
              disabled={generating}
            />
            <button className="btn btn-send" onClick={sendMessage} disabled={generating}>
              Send
            </button>
          </div>
        </section>

        {/* Playlist Panel */}
        <section className="playlist-panel">
          {currentPlaylist ? (
            <>
              <div className="panel-header">
                <div className="tab-bar">
                  <button
                    className={`tab ${activePlaylist === 'a' ? 'active' : ''}`}
                    onClick={() => setActivePlaylist('a')}
                  >
                    Playlist A
                    {playlists.a && <span className="tab-score">{playlists.a.scores.total.toFixed(2)}</span>}
                  </button>
                  <button
                    className={`tab ${activePlaylist === 'b' ? 'active' : ''}`}
                    onClick={() => setActivePlaylist('b')}
                  >
                    Playlist B
                    {playlists.b && <span className="tab-score">{playlists.b.scores.total.toFixed(2)}</span>}
                  </button>
                </div>
              </div>

              {/* Score Gauges */}
              <div className="score-bar">
                {['fit', 'flow', 'variety', 'accessibility', 'quality'].map(dim => (
                  <div key={dim} className="score-item">
                    <div className="score-label">{dim}</div>
                    <div className="score-track">
                      <div
                        className="score-fill"
                        style={{ width: `${(currentPlaylist.scores[dim] || 0) * 100}%` }}
                      />
                    </div>
                    <div className="score-value">{((currentPlaylist.scores[dim] || 0) * 100).toFixed(0)}%</div>
                  </div>
                ))}
              </div>

              <div className="playlist-meta">
                <span>{currentPlaylist.tracks.length} tracks</span>
                <span>·</span>
                <span>{formatTotalDuration(currentPlaylist.total_duration_s)}</span>
                <span>·</span>
                <span className={`total-score ${currentPlaylist.scores.total >= 0.8 ? 'score-good' : currentPlaylist.scores.total >= 0.6 ? 'score-ok' : 'score-low'}`}>
                  Score: {currentPlaylist.scores.total.toFixed(2)}
                </span>
              </div>

              {/* Track List with Drag & Drop */}
              <DndContext
                sensors={sensors}
                collisionDetection={closestCenter}
                onDragEnd={handleDragEnd}
              >
                <SortableContext
                  items={currentPlaylist.tracks.map(t => t.id)}
                  strategy={verticalListSortingStrategy}
                >
                  <div className="track-list">
                    {currentPlaylist.tracks.map((track, i) => (
                      <SortableTrack
                        key={track.id}
                        track={track}
                        index={i}
                        onRemove={removeTrack}
                        formatDuration={formatDuration}
                      />
                    ))}
                  </div>
                </SortableContext>
              </DndContext>
            </>
          ) : (
            <div className="empty-state">
              <div className="empty-icon">🎧</div>
              <h3>No playlist yet</h3>
              <p>Chat with the AI curator to generate your perfect mixtape.</p>
            </div>
          )}
        </section>
      </main>

      {/* Search Modal */}
      {showSearch && (
        <div className="modal-overlay" onClick={() => setShowSearch(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Search Library</h3>
              <button className="btn-icon" onClick={() => setShowSearch(false)}>✕</button>
            </div>
            <div className="search-bar">
              <input
                type="text"
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && searchLibrary()}
                placeholder="Search by artist, genre, or descriptor..."
                autoFocus
              />
              <button className="btn btn-send" onClick={searchLibrary}>Search</button>
            </div>
            <div className="search-results">
              {searchResults.map(track => (
                <div key={track.id} className="search-result" onClick={() => addTrack(track.id)}>
                  <div className="track-info">
                    <div className="track-title">{track.title}</div>
                    <div className="track-artist">{track.artist}</div>
                  </div>
                  <div className="track-tags">
                    {track.genres.map(g => <span key={g} className="tag tag-genre">{g}</span>)}
                  </div>
                  <div className="track-duration">{formatDuration(track.duration_s)}</div>
                  <button className="btn btn-outline btn-sm">+ Add</button>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Label Designer Modal */}
      {showLabel && (
        <div className="modal-overlay" onClick={() => setShowLabel(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>💿 CD Label Designer</h3>
              <button className="btn-icon" onClick={() => setShowLabel(false)}>✕</button>
            </div>
            <div className="label-form">
              <label>
                Mixtape Title
                <input
                  type="text"
                  value={labelTitle}
                  onChange={e => setLabelTitle(e.target.value)}
                  placeholder="My Mixtape"
                />
              </label>
              <label>
                Cover Art Description (optional — uses AI to generate)
                <textarea
                  value={labelPrompt}
                  onChange={e => setLabelPrompt(e.target.value)}
                  placeholder="e.g. 'Retro 80s neon sunset, synthwave vibes, dark purple and pink'"
                  rows={3}
                />
              </label>
              <p className="label-hint">
                Leave the description empty for a minimal dark label, or describe the vibe and AI will generate cover art.
              </p>
              <button className="btn btn-spotify" onClick={downloadLabel}>
                📥 Generate & Download PDF
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
