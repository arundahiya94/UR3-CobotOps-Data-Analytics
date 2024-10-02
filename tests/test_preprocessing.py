import unittest
import pandas as pd
from src.preprocessing import clean_data

class TestPreprocessing(unittest.TestCase):

    def test_clean_data(self):
        """Test if clean_data correctly drops NaNs and unnecessary columns."""
        data = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [None, 2, 3, 4],
            'Timestamp': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']
        })
        cleaned_data = clean_data(data)
        self.assertEqual(cleaned_data.isna().sum().sum(), 0)
        self.assertNotIn('Timestamp', cleaned_data.columns)

if __name__ == '__main__':
    unittest.main()
