from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

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
    
    # Metadata features
    accessibility: float = 0.0
    familiarity: float = 0.0
    rating: float = 0.0
    recommendability: float = 0.0
    
    # System fields
    file_path: Optional[str] = None
    spotify_uri: Optional[str] = None

class FeedbackTargets(BaseModel):
    uniformity: float = 0.5  # 0.0 = Eclectic, 1.0 = Uniform
    energy: float = 0.5
    valence: float = 0.5
    intensity: float = 0.5
    accessibility: float = 0.5
    familiarity: float = 0.5

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
