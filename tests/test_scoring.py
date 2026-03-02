import pytest
from mixtape_curator.scoring import scorer
from mixtape_curator.models import Track, UserProfile, RYMData

@pytest.fixture
def mock_track():
    return Track(
        id="t1", title="Test", artist="Art", duration_s=100,
        energy=0.8, valence=0.8, intensity=0.8,
        rym_data=RYMData(primary_genres=["Pop"])
    )

@pytest.fixture
def mock_profile():
    p = UserProfile()
    p.targets.energy = 0.8
    p.targets.valence = 0.8
    p.targets.intensity = 0.8
    p.target_genres = ["Pop"]
    p.targets.uniformity = 0.0 # Eclectic -> High Diversity
    return p

def test_fit_score_perfect(mock_track, mock_profile):
    score = scorer.compute_fit_score([mock_track], mock_profile)
    assert score >= 0.99

def test_fit_score_decay(mock_track, mock_profile):
    # Move target away
    mock_profile.targets.energy = 0.0 
    score = scorer.compute_fit_score([mock_track], mock_profile)
    # Blended with genre fit (1.0) at 0.3 strictness
    assert 0.2 < score < 0.4

def test_variety_score_uniformity(mock_track, mock_profile):
    # 3 identical tracks -> Low diversity
    tracks = [mock_track, mock_track, mock_track]
    
    # Profile wants HIGH diversity (uniformity=0.0) -> ideal_diversity=1.0
    # Measured diversity ~ 0.0
    # Genre variety = 1 - abs(0 - 1) = 0.0
    # Artist variety = 1/3 = 0.33
    # Final = 0.7 * 0.0 + 0.3 * 0.33 = ~0.1
    score_eclectic = scorer.compute_variety_score(tracks, mock_profile)
    assert score_eclectic < 0.2  # Adjusted threshold for artist variety component
    
    # Profile wants LOW diversity (uniformity=1.0) -> ideal_diversity=0.0
    # Measured diversity ~ 0.0 (all same genre)
    # Genre variety = 1 - abs(0 - 0) = 1.0
    # Artist variety = 1/3 = 0.33
    # Final = 0.7 * 1.0 + 0.3 * 0.33 = 0.8
    mock_profile.targets.uniformity = 1.0
    score_uniform = scorer.compute_variety_score(tracks, mock_profile)
    assert score_uniform > 0.7  # Adjusted to account for artist variety penalty

def test_quality_liked_boost(mock_profile):
    """A liked track should score higher than an identical non-liked track."""
    base = Track(
        id="t_base", title="Base", artist="Art", duration_s=200,
        rating=0.7, recommendability=0.5, liked=False,
        familiarity=0.5, mix_prominence=0.0, artist_mix_prominence=0.0,
        spotify_affinity=0.0,
        rym_data=RYMData(primary_genres=["Rock"])
    )
    liked = base.model_copy(update={"id": "t_liked", "liked": True})

    scores_base = scorer.score_playlist([base], mock_profile)
    scores_liked = scorer.score_playlist([liked], mock_profile)

    assert scores_liked.quality > scores_base.quality
    # The gap should be exactly 0.30 (liked_bonus weight)
    assert abs((scores_liked.quality - scores_base.quality) - 0.30) < 0.01

def test_quality_not_liked_baseline(mock_profile):
    """Non-liked track quality = rating*0.30 + mix*0.25 + spotify*0.15."""
    track = Track(
        id="t1", title="Test", artist="Art", duration_s=200,
        rating=0.8, recommendability=0.5, liked=False,
        familiarity=0.4, mix_prominence=0.3, artist_mix_prominence=0.2,
        spotify_affinity=0.5,
        rym_data=RYMData(primary_genres=["Pop"])
    )
    scores = scorer.score_playlist([track], mock_profile)
    expected_quality = (0.8*0.30) + (0.0*0.30) + (0.3*0.25) + (0.5*0.15)
    assert abs(scores.quality - expected_quality) < 0.01

def test_quality_mix_prominence_boost(mock_profile):
    """A track with high mix_prominence should score higher than one without."""
    base = Track(
        id="t_base", title="Base", artist="Art", duration_s=200,
        rating=0.7, liked=False, familiarity=0.5,
        mix_prominence=0.0, artist_mix_prominence=0.0, spotify_affinity=0.0,
        rym_data=RYMData(primary_genres=["Rock"])
    )
    prominent = base.model_copy(update={"id": "t_prom", "mix_prominence": 1.0})
    scores_base = scorer.score_playlist([base], mock_profile)
    scores_prom = scorer.score_playlist([prominent], mock_profile)
    assert scores_prom.quality > scores_base.quality
    # Gap should be exactly 0.25 (mix_prominence weight)
    assert abs((scores_prom.quality - scores_base.quality) - 0.25) < 0.01


