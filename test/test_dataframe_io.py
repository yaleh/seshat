import unittest
import tempfile
import os
import pandas as pd
from unittest.mock import patch, MagicMock

from tools.dataframe_io import dump_dataframe


class TestDumpDataframe(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_df = pd.DataFrame({
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'City': ['New York', 'London', 'Tokyo']
        })
        self.temp_files = []
    
    def tearDown(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_dump_dataframe_csv(self):
        """Test dumping DataFrame to CSV"""
        file_types = ['csv']
        
        result = dump_dataframe(self.test_df, file_types)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].endswith('.csv'))
        self.assertTrue(os.path.exists(result[0]))
        
        # Verify content
        loaded_df = pd.read_csv(result[0])
        pd.testing.assert_frame_equal(loaded_df, self.test_df)
        
        self.temp_files.extend(result)
    
    def test_dump_dataframe_xlsx(self):
        """Test dumping DataFrame to Excel"""
        file_types = ['xlsx']
        
        result = dump_dataframe(self.test_df, file_types)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].endswith('.xlsx'))
        self.assertTrue(os.path.exists(result[0]))
        
        # Verify content
        loaded_df = pd.read_excel(result[0])
        pd.testing.assert_frame_equal(loaded_df, self.test_df)
        
        self.temp_files.extend(result)
    
    def test_dump_dataframe_multiple_formats(self):
        """Test dumping DataFrame to multiple formats"""
        file_types = ['csv', 'xlsx']
        
        result = dump_dataframe(self.test_df, file_types)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        
        # Check that we got both formats
        extensions = [os.path.splitext(f)[1] for f in result]
        self.assertIn('.csv', extensions)
        self.assertIn('.xlsx', extensions)
        
        # Verify both files exist
        for file_path in result:
            self.assertTrue(os.path.exists(file_path))
        
        self.temp_files.extend(result)
    
    def test_dump_dataframe_none_input(self):
        """Test with None DataFrame"""
        file_types = ['csv']
        
        result = dump_dataframe(None, file_types)
        
        self.assertIsNone(result)
    
    def test_dump_dataframe_empty_dataframe(self):
        """Test with empty DataFrame"""
        empty_df = pd.DataFrame()
        file_types = ['csv']
        
        result = dump_dataframe(empty_df, file_types)
        
        self.assertIsNone(result)
    
    def test_dump_dataframe_single_row(self):
        """Test with single row DataFrame (should return None)"""
        single_row_df = pd.DataFrame({'Name': ['Alice'], 'Age': [25]})
        file_types = ['csv']
        
        result = dump_dataframe(single_row_df, file_types)
        
        self.assertIsNone(result)
    
    def test_dump_dataframe_two_rows(self):
        """Test with two rows DataFrame (should work)"""
        two_row_df = pd.DataFrame({
            'Name': ['Alice', 'Bob'], 
            'Age': [25, 30]
        })
        file_types = ['csv']
        
        result = dump_dataframe(two_row_df, file_types)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        
        self.temp_files.extend(result)
    
    def test_dump_dataframe_unsupported_format(self):
        """Test with unsupported file format"""
        file_types = ['json', 'xml']  # Unsupported formats
        
        result = dump_dataframe(self.test_df, file_types)
        
        # Should return empty list since no supported formats
        self.assertEqual(result, [])
    
    def test_dump_dataframe_mixed_supported_unsupported(self):
        """Test with mix of supported and unsupported formats"""
        file_types = ['csv', 'json', 'xlsx', 'xml']  # Mix of supported and unsupported
        
        result = dump_dataframe(self.test_df, file_types)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)  # Only CSV and XLSX should be created
        
        extensions = [os.path.splitext(f)[1] for f in result]
        self.assertIn('.csv', extensions)
        self.assertIn('.xlsx', extensions)
        
        self.temp_files.extend(result)
    
    def test_dump_dataframe_empty_file_types(self):
        """Test with empty file_types list"""
        file_types = []
        
        result = dump_dataframe(self.test_df, file_types)
        
        self.assertEqual(result, [])
    
    @patch('tools.dataframe_io.tempfile.NamedTemporaryFile')
    def test_dump_dataframe_csv_temp_file_creation(self, mock_temp_file):
        """Test CSV temporary file creation process"""
        mock_file = MagicMock()
        mock_file.name = '/tmp/test.csv'
        mock_temp_file.return_value.__enter__.return_value = mock_file
        
        # Mock DataFrame.to_csv
        with patch.object(self.test_df, 'to_csv') as mock_to_csv:
            result = dump_dataframe(self.test_df, ['csv'])
            
            mock_temp_file.assert_called_once_with(suffix=".csv", delete=False)
            mock_to_csv.assert_called_once_with('/tmp/test.csv', index=False)
            self.assertEqual(result, ['/tmp/test.csv'])
    
    @patch('tools.dataframe_io.tempfile.NamedTemporaryFile')
    def test_dump_dataframe_xlsx_temp_file_creation(self, mock_temp_file):
        """Test Excel temporary file creation process"""
        mock_file = MagicMock()
        mock_file.name = '/tmp/test.xlsx'
        mock_temp_file.return_value.__enter__.return_value = mock_file
        
        # Mock DataFrame.to_excel
        with patch.object(self.test_df, 'to_excel') as mock_to_excel:
            result = dump_dataframe(self.test_df, ['xlsx'])
            
            mock_temp_file.assert_called_once_with(suffix=".xlsx", delete=False)
            mock_to_excel.assert_called_once_with('/tmp/test.xlsx', index=False)
            self.assertEqual(result, ['/tmp/test.xlsx'])
    
    def test_dump_dataframe_with_special_characters(self):
        """Test with DataFrame containing special characters"""
        special_df = pd.DataFrame({
            'Name': ['测试', 'тест', '🔥'],
            'Description': ['Special chars: àáâãäå', 'Unicode: ñüáéíóú', 'Emoji: 😀🎉']
        })
        file_types = ['csv']
        
        result = dump_dataframe(special_df, file_types)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        
        # Verify file was created and can be read back
        loaded_df = pd.read_csv(result[0])
        self.assertEqual(loaded_df.shape, special_df.shape)
        
        self.temp_files.extend(result)


if __name__ == '__main__':
    unittest.main()