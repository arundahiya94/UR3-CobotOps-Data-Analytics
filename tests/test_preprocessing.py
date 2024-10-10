import unittest
from src.preprocessing import clean_data, normalize_data
import pandas as pd

class TestPreprocessing(unittest.TestCase):
    
    def test_clean_data(self):
        df = pd.DataFrame({'grip_lost': ['1', '0'], 'Robot_ProtectiveStop': ['1', '0'], 'Temperature': [20, None]})
        clean_df = clean_data(df)
        self.assertEqual(len(clean_df), 1)

    def test_normalize_data(self):
        df = pd.DataFrame({'Temperature': [20, 30]})
        norm_df = normalize_data(df, ['Temperature'])
        self.assertTrue(norm_df['Temperature'].max() <= 1)
        self.assertTrue(norm_df['Temperature'].min() >= 0)

if __name__ == '__main__':
    unittest.main()
