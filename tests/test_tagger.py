
import unittest
from unittest.mock import patch, MagicMock
from mixtape_curator.library import library
from mixtape_curator.metadata_tagger import MetadataTagger
import pandas as pd

class TestTagger(unittest.TestCase):
    def setUp(self):
        # Mock dataframe
        self.df = pd.DataFrame([
            {'id': 't1', 'title': 'One', 'artist': 'A', 'release_year': None, 'energy': 0.1},
            {'id': 't2', 'title': 'Two', 'artist': 'B', 'release_year': 1999, 'energy': 0.5}
        ])
        library.df = self.df
        
    def test_fill_missing_year(self):
        tagger = MetadataTagger()
        
        # Simulating one input '2000' then 'q' to quit
        with patch('builtins.input', side_effect=['2000', 'q']):
            try:
                tagger.do_fill_years("")
            except Exception as e:
                pass
                
        # Verify first row updated
        self.assertEqual(library.df.iloc[0]['release_year'], 2000)
        # Verify second row untouched
        self.assertEqual(library.df.iloc[1]['release_year'], 1999)

    def test_rate_energy(self):
        tagger = MetadataTagger()
        
        # User sets Energy=0.9 for first track, keeps Valence same (skip)
        with patch('builtins.input', side_effect=['0.9', '', 'q']):
            try:
                tagger.do_rate_vibe("")
            except: pass
            
        self.assertAlmostEqual(library.df.iloc[0]['energy'], 0.9)

if __name__ == '__main__':
    unittest.main()
