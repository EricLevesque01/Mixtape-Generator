import pytest
from src.mixtape_curator.scoring import scorer
from src.mixtape_curator.models import Track, UserProfile, RYMData

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
    # Distance = 0.8^2 = 0.64. 
    # d^2 total = 0.64. sonic_fit = exp(-0.64 / (2 * 0.25^2)) = exp(-0.64/0.125) = exp(-5.12) ~= 0.006
    score = scorer.compute_fit_score([mock_track], mock_profile)
    # Blended with genre fit (1.0) at 0.3 strictness
    # 0.006 * 0.7 + 1.0 * 0.3 ~= 0.304
    assert 0.2 < score < 0.4

def test_variety_score_uniformity(mock_track, mock_profile):
    # 3 identical tracks -> Low diversity
    tracks = [mock_track, mock_track, mock_track]
    
    # Profile wants HIGH diversity (uniformity=0.0) -> ideal_diversity=1.0
    # Measured diversity ~ 0.0
    # Score = 1 - abs(0 - 1) = 0.0
    score_eclectic = scorer.compute_variety_score(tracks, mock_profile)
    assert score_eclectic < 0.1
    
    # Profile wants LOW diversity (uniformity=1.0) -> ideal_diversity=0.0
    mock_profile.targets.uniformity = 1.0
    score_uniform = scorer.compute_variety_score(tracks, mock_profile)
    assert score_uniform > 0.9
