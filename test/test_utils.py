import unittest
import tempfile
import os
from unittest.mock import patch, mock_open, MagicMock

from tools.utils import detect_encoding


class TestDetectEncoding(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file_path = self.temp_file.name
    
    def tearDown(self):
        """Clean up temp files"""
        if os.path.exists(self.temp_file_path):
            os.unlink(self.temp_file_path)
    
    @patch('tools.utils.chardet.detect')
    @patch('builtins.open', new_callable=mock_open, read_data=b'test data')
    def test_detect_encoding_utf8(self, mock_file, mock_chardet):
        """Test detecting UTF-8 encoding"""
        mock_chardet.return_value = {'encoding': 'utf-8', 'confidence': 0.99}
        
        result = detect_encoding(self.temp_file_path)
        
        self.assertEqual(result, 'utf-8')
        mock_file.assert_called_once_with(self.temp_file_path, 'rb')
        mock_chardet.assert_called_once()
    
    @patch('tools.utils.chardet.detect')
    @patch('builtins.open', new_callable=mock_open, read_data=b'test data')
    def test_detect_encoding_iso_8859_1_returns_gbk(self, mock_file, mock_chardet):
        """Test that ISO-8859-1 encoding is mapped to GBK"""
        mock_chardet.return_value = {'encoding': 'ISO-8859-1', 'confidence': 0.85}
        
        result = detect_encoding(self.temp_file_path)
        
        self.assertEqual(result, 'gbk')
        mock_file.assert_called_once_with(self.temp_file_path, 'rb')
        mock_chardet.assert_called_once()
    
    @patch('tools.utils.chardet.detect')
    @patch('builtins.open', new_callable=mock_open, read_data=b'test data')
    def test_detect_encoding_ascii(self, mock_file, mock_chardet):
        """Test detecting ASCII encoding"""
        mock_chardet.return_value = {'encoding': 'ascii', 'confidence': 1.0}
        
        result = detect_encoding(self.temp_file_path)
        
        self.assertEqual(result, 'ascii')
        mock_file.assert_called_once_with(self.temp_file_path, 'rb')
        mock_chardet.assert_called_once()
    
    @patch('tools.utils.chardet.detect')
    @patch('builtins.open', new_callable=mock_open, read_data=b'\xe4\xb8\xad\xe6\x96\x87')
    def test_detect_encoding_with_chinese_content(self, mock_file, mock_chardet):
        """Test detecting encoding with Chinese content"""
        mock_chardet.return_value = {'encoding': 'utf-8', 'confidence': 0.99}
        
        result = detect_encoding(self.temp_file_path)
        
        self.assertEqual(result, 'utf-8')
        mock_chardet.assert_called_once_with(b'\xe4\xb8\xad\xe6\x96\x87')
    
    @patch('tools.utils.chardet.detect')
    @patch('builtins.open', new_callable=mock_open, read_data=b'')
    def test_detect_encoding_empty_file(self, mock_file, mock_chardet):
        """Test detecting encoding of empty file"""
        mock_chardet.return_value = {'encoding': None, 'confidence': 0.0}
        
        result = detect_encoding(self.temp_file_path)
        
        self.assertIsNone(result)
        mock_chardet.assert_called_once_with(b'')
    
    def test_detect_encoding_real_file_utf8(self):
        """Test with real UTF-8 file"""
        # Create a real file with UTF-8 content
        with open(self.temp_file_path, 'w', encoding='utf-8') as f:
            f.write('Hello World! 你好世界!')
        
        result = detect_encoding(self.temp_file_path)
        
        # Should detect UTF-8 or compatible encoding
        self.assertIsNotNone(result)
        self.assertIn(result.lower(), ['utf-8', 'utf8'])
    
    def test_detect_encoding_real_file_ascii(self):
        """Test with real ASCII file"""
        # Create a real file with ASCII content
        with open(self.temp_file_path, 'w', encoding='ascii') as f:
            f.write('Hello World!')
        
        result = detect_encoding(self.temp_file_path)
        
        # Should detect ASCII or compatible encoding
        self.assertIsNotNone(result)
        self.assertIn(result.lower(), ['ascii', 'utf-8', 'utf8'])
    
    def test_detect_encoding_file_not_found(self):
        """Test with non-existent file"""
        with self.assertRaises(FileNotFoundError):
            detect_encoding('non_existent_file.txt')


if __name__ == '__main__':
    unittest.main()