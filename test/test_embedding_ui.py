import unittest
import pandas as pd
import tempfile
import json
from unittest.mock import patch, MagicMock, Mock
import gradio as gr

from ui.embedding_ui import EmbeddingUI, model_settings


class TestEmbeddingUIBasics(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        self.mock_config.server.message_db = 'test.db'
        self.mock_config.server.max_message_length = 1000
        # Mock config.embedding to have objects with .type and .dict() method
        mock_embedding_config = MagicMock()
        mock_embedding_config.type = 'OpenAI'
        mock_embedding_config.dict.return_value = {'openai_api_key': 'test-key', 'type': 'OpenAI'}
        self.mock_config.embedding = {
            'text-embedding-ada-002': mock_embedding_config,
            'text-embedding-3-small': mock_embedding_config
        }
    
    @patch('ui.embedding_ui.EmbeddingUI.init_ui')
    @patch('ui.embedding_ui.DatabaseManager')
    def test_embedding_ui_init_basics(self, mock_db_manager, mock_init_ui):
        """Test EmbeddingUI basic initialization without UI creation"""
        # Setup mocks
        mock_db_instance = MagicMock()
        mock_db_manager.return_value = mock_db_instance
        
        mock_ui_instance = MagicMock()
        mock_init_ui.return_value = mock_ui_instance
        
        # Create EmbeddingUI
        embedding_ui = EmbeddingUI(self.mock_config)
        
        # Verify initialization
        self.assertEqual(embedding_ui.config, self.mock_config)
        mock_db_manager.assert_called_once_with('test.db', 1000)
        self.assertEqual(embedding_ui.db_manager, mock_db_instance)
        
        # Verify VDB settings initialization
        self.assertIn('Pinecone', embedding_ui.vdb_settings)
        self.assertIn('Milvus', embedding_ui.vdb_settings)
        self.assertIn('Chroma', embedding_ui.vdb_settings)
        self.assertEqual(embedding_ui.current_vdb_type, 'Pinecone')
        
        # Verify UI was created
        self.assertEqual(embedding_ui.ui, mock_ui_instance)
    
    @patch('ui.embedding_ui.DatabaseManager')
    def test_embedding_ui_init_with_none_config(self, mock_db_manager):
        """Test EmbeddingUI initialization with None config raises AttributeError"""
        with self.assertRaises(AttributeError):
            EmbeddingUI(None)
    
    def test_model_settings_global_variable(self):
        """Test that model_settings global variable is properly defined"""
        self.assertIsInstance(model_settings, list)
        self.assertGreater(len(model_settings), 0)
        
        for setting in model_settings:
            self.assertIn('model', setting)
            self.assertIsInstance(setting['model'], str)


class TestEmbeddingUITextProcessing(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        self.mock_config.server.message_db = 'test.db'
        self.mock_config.server.max_message_length = 1000
        # Mock config.embedding to have objects with .type and .dict() method
        mock_embedding_config = MagicMock()
        mock_embedding_config.type = 'OpenAI'
        mock_embedding_config.dict.return_value = {'openai_api_key': 'test-key', 'type': 'OpenAI'}
        self.mock_config.embedding = {
            'text-embedding-ada-002': mock_embedding_config
        }
    
    @patch('ui.embedding_ui.EmbeddingUI.init_ui')
    @patch('ui.embedding_ui.DatabaseManager')
    @patch('ui.embedding_ui.EmbeddingFactory')
    def test_create_embedding_model(self, mock_embedding_factory, mock_db_manager, mock_init_ui):
        """Test _create_embedding_model method"""
        # Setup mocks
        mock_db_manager.return_value = MagicMock()
        mock_init_ui.return_value = MagicMock()
        
        mock_embedding_instance = MagicMock()
        mock_embedding_factory.return_value.create.return_value = mock_embedding_instance
        
        # Create EmbeddingUI
        embedding_ui = EmbeddingUI(self.mock_config)
        
        # Test _create_embedding_model
        result = embedding_ui._create_embedding_model('text-embedding-ada-002')
        
        # Verify embedding factory was called correctly
        mock_embedding_factory.assert_called_once()
        factory_instance = mock_embedding_factory.return_value
        factory_instance.create.assert_called_once_with(
            'OpenAI',
            openai_api_key='test-key'
        )
        
        self.assertEqual(result, mock_embedding_instance)
    
    @patch('ui.embedding_ui.EmbeddingUI.init_ui')
    @patch('ui.embedding_ui.DatabaseManager')
    @patch('ui.embedding_ui.EmbeddingUI._create_embedding_model')
    def test_embed_text_success(self, mock_create_model, mock_db_manager, mock_init_ui):
        """Test embed_text method with successful embedding"""
        # Setup mocks
        mock_db_manager.return_value = MagicMock()
        mock_init_ui.return_value = MagicMock()
        
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_create_model.return_value = mock_embedding_model
        
        # Create EmbeddingUI
        embedding_ui = EmbeddingUI(self.mock_config)
        
        # Test embed_text
        result = embedding_ui.embed_text("test text", "text-embedding-ada-002")
        
        # Verify model creation and embedding
        mock_create_model.assert_called_once_with('text-embedding-ada-002')
        mock_embedding_model.embed_query.assert_called_once_with("test text")
        
        # embed_text returns the raw embedding vector, not JSON
        expected_result = [0.1, 0.2, 0.3]
        self.assertEqual(result, expected_result)
    
    @patch('ui.embedding_ui.EmbeddingUI.init_ui')
    @patch('ui.embedding_ui.DatabaseManager')
    @patch('ui.embedding_ui.EmbeddingUI._create_embedding_model')
    def test_embed_text_failure(self, mock_create_model, mock_db_manager, mock_init_ui):
        """Test embed_text method with embedding failure"""
        # Setup mocks
        mock_db_manager.return_value = MagicMock()
        mock_init_ui.return_value = MagicMock()
        
        mock_create_model.side_effect = Exception("Embedding failed")
        
        # Create EmbeddingUI
        embedding_ui = EmbeddingUI(self.mock_config)
        
        # Test embed_text with exception - it should raise gr.Error
        with patch('ui.embedding_ui.gr.Error', side_effect=Exception) as mock_gr_error:
            with self.assertRaises(Exception):
                embedding_ui.embed_text("test text", "text-embedding-ada-002")
            
            # Verify gr.Error was called
            mock_gr_error.assert_called_once()


class TestEmbeddingUIDataFrameProcessing(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        self.mock_config.server.message_db = 'test.db'
        self.mock_config.server.max_message_length = 1000
        # Mock config.embedding to have objects with .type and .dict() method
        mock_embedding_config = MagicMock()
        mock_embedding_config.type = 'OpenAI'
        mock_embedding_config.dict.return_value = {'openai_api_key': 'test-key', 'type': 'OpenAI'}
        self.mock_config.embedding = {
            'text-embedding-ada-002': mock_embedding_config
        }
    
    @patch('ui.embedding_ui.EmbeddingUI.init_ui')
    @patch('ui.embedding_ui.DatabaseManager')
    def test_upload_data_csv(self, mock_db_manager, mock_init_ui):
        """Test upload_data method with CSV file"""
        # Setup mocks
        mock_db_manager.return_value = MagicMock()
        mock_init_ui.return_value = MagicMock()
        
        # Create EmbeddingUI
        embedding_ui = EmbeddingUI(self.mock_config)
        
        # Create a mock file object
        mock_file = MagicMock()
        mock_file.name = 'test.csv'
        
        with patch('ui.embedding_ui.detect_encoding', return_value='utf-8'), \
             patch('pandas.read_csv') as mock_read_csv:
            
            test_df = pd.DataFrame({'id': [1, 2], 'text': ['hello', 'world']})
            mock_read_csv.return_value = test_df
            
            result = embedding_ui.upload_data(mock_file)
            
            # Verify file was read correctly
            mock_read_csv.assert_called_once_with('test.csv', encoding='utf-8')
            
            # upload_data returns a tuple: (df, key_field, value_fields, batch_start_update, batch_end_update, batch_size_update)
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 6)
            pd.testing.assert_frame_equal(result[0], test_df)  # First element is the DataFrame
    
    @patch('ui.embedding_ui.EmbeddingUI.init_ui')
    @patch('ui.embedding_ui.DatabaseManager')
    def test_upload_data_excel(self, mock_db_manager, mock_init_ui):
        """Test upload_data method with Excel file"""
        # Setup mocks
        mock_db_manager.return_value = MagicMock()
        mock_init_ui.return_value = MagicMock()
        
        # Create EmbeddingUI
        embedding_ui = EmbeddingUI(self.mock_config)
        
        # Create a mock file object
        mock_file = MagicMock()
        mock_file.name = 'test.xlsx'
        
        with patch('pandas.read_excel') as mock_read_excel:
            test_df = pd.DataFrame({'id': [1, 2], 'text': ['hello', 'world']})
            mock_read_excel.return_value = test_df
            
            result = embedding_ui.upload_data(mock_file)
            
            # Verify file was read correctly
            mock_read_excel.assert_called_once_with('test.xlsx')
            
            # upload_data returns a tuple: (df, key_field, value_fields, batch_start_update, batch_end_update, batch_size_update)
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 6)
            pd.testing.assert_frame_equal(result[0], test_df)  # First element is the DataFrame
    
    @patch('ui.embedding_ui.EmbeddingUI.init_ui')
    @patch('ui.embedding_ui.DatabaseManager')
    def test_upload_data_invalid_file(self, mock_db_manager, mock_init_ui):
        """Test upload_data method with unsupported file format"""
        # Setup mocks
        mock_db_manager.return_value = MagicMock()
        mock_init_ui.return_value = MagicMock()
        
        # Create EmbeddingUI
        embedding_ui = EmbeddingUI(self.mock_config)
        
        # Create a mock file object with unsupported format
        mock_file = MagicMock()
        mock_file.name = 'test.txt'  # Unsupported format
        
        result = embedding_ui.upload_data(mock_file)
        
        # Verify tuple was returned with empty DataFrame (no error raised for unsupported format)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 6)
        self.assertTrue(result[0].empty)  # First element should be empty DataFrame
        self.assertEqual(result[1], "id")  # key_field default
        self.assertEqual(result[2], "text")  # value_fields default
    
    @patch('ui.embedding_ui.EmbeddingUI.init_ui')
    @patch('ui.embedding_ui.DatabaseManager')
    def test_save_table(self, mock_db_manager, mock_init_ui):
        """Test save_table method"""
        # Setup mocks
        mock_db_manager.return_value = MagicMock()
        mock_init_ui.return_value = MagicMock()
        
        # Create EmbeddingUI
        embedding_ui = EmbeddingUI(self.mock_config)
        
        # Create test dataframe
        test_df = pd.DataFrame({'id': [1, 2], 'text': ['hello', 'world']})
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
            # Mock NamedTemporaryFile to return an object with .name attribute
            mock_file = MagicMock()
            mock_file.name = '/tmp/test.csv'
            mock_temp_file.return_value = mock_file
            
            with patch.object(test_df, 'to_csv') as mock_to_csv:
                result = embedding_ui.save_table(test_df)
                
                # Verify tempfile was called correctly
                mock_temp_file.assert_called_once_with(suffix=".csv", delete=False)
                
                # Verify CSV was saved with the mock file name
                mock_to_csv.assert_called_once_with('/tmp/test.csv', index=False)
                
                # Verify the filename is returned
                self.assertEqual(result, '/tmp/test.csv')
    
    @patch('ui.embedding_ui.EmbeddingUI.init_ui')
    @patch('ui.embedding_ui.DatabaseManager')
    def test_add_index_column(self, mock_db_manager, mock_init_ui):
        """Test add_index_column method"""
        # Setup mocks
        mock_db_manager.return_value = MagicMock()
        mock_init_ui.return_value = MagicMock()
        
        # Create EmbeddingUI
        embedding_ui = EmbeddingUI(self.mock_config)
        
        # Create test dataframe
        test_df = pd.DataFrame({'text': ['hello', 'world']})
        
        result = embedding_ui.add_index_column(test_df)
        
        # Verify index column was added
        self.assertIn('index', result.columns)
        self.assertEqual(list(result['index']), [0, 1])
        self.assertEqual(list(result['text']), ['hello', 'world'])


class TestEmbeddingUIVDBSettings(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        self.mock_config.server.message_db = 'test.db'
        self.mock_config.server.max_message_length = 1000
        # Mock config.embedding to have objects with .type and .dict() method
        mock_embedding_config = MagicMock()
        mock_embedding_config.type = 'OpenAI'
        mock_embedding_config.dict.return_value = {'openai_api_key': 'test-key', 'type': 'OpenAI'}
        self.mock_config.embedding = {
            'text-embedding-ada-002': mock_embedding_config
        }
    
    @patch('ui.embedding_ui.EmbeddingUI.init_ui')
    @patch('ui.embedding_ui.DatabaseManager')
    def test_update_pinecone_settings(self, mock_db_manager, mock_init_ui):
        """Test update_pinecone_settings method"""
        # Setup mocks
        mock_db_instance = MagicMock()
        mock_db_manager.return_value = mock_db_instance
        mock_init_ui.return_value = MagicMock()
        
        # Create EmbeddingUI
        embedding_ui = EmbeddingUI(self.mock_config)
        
        # Test update_pinecone_settings
        embedding_ui.update_pinecone_settings("test-host", "test-api-key")
        
        # Verify settings were updated
        self.assertEqual(embedding_ui.vdb_settings["Pinecone"]["host"], "test-host")
        self.assertEqual(embedding_ui.vdb_settings["Pinecone"]["api_key"], "test-api-key")
        
        # update_pinecone_settings only updates the vdb_settings dictionary, not the database
        # Database updates happen in update_vdb_history_db method
        mock_db_instance.append_message.assert_not_called()
    
    @patch('ui.embedding_ui.EmbeddingUI.init_ui')
    @patch('ui.embedding_ui.DatabaseManager')
    def test_update_milvus_settings(self, mock_db_manager, mock_init_ui):
        """Test update_milvus_settings method"""
        # Setup mocks
        mock_db_instance = MagicMock()
        mock_db_manager.return_value = mock_db_instance
        mock_init_ui.return_value = MagicMock()
        
        # Create EmbeddingUI
        embedding_ui = EmbeddingUI(self.mock_config)
        
        # Test update_milvus_settings
        embedding_ui.update_milvus_settings("test-uri", "test-token", "test-collection")
        
        # Verify settings were updated
        self.assertEqual(embedding_ui.vdb_settings["Milvus"]["uri"], "test-uri")
        self.assertEqual(embedding_ui.vdb_settings["Milvus"]["token"], "test-token")
        self.assertEqual(embedding_ui.vdb_settings["Milvus"]["collection_name"], "test-collection")
        
        # update_milvus_settings only updates the vdb_settings dictionary, not the database
        # Database updates happen in update_vdb_history_db method
        mock_db_instance.append_message.assert_not_called()
    
    @patch('ui.embedding_ui.EmbeddingUI.init_ui')
    @patch('ui.embedding_ui.DatabaseManager')
    def test_select_vdb_tab(self, mock_db_manager, mock_init_ui):
        """Test select_vdb_tab method"""
        # Setup mocks
        mock_db_manager.return_value = MagicMock()
        mock_init_ui.return_value = MagicMock()
        
        # Create EmbeddingUI
        embedding_ui = EmbeddingUI(self.mock_config)
        
        # Test select_vdb_tab with mock event
        mock_event = MagicMock()
        mock_event.value = "Milvus"
        
        result = embedding_ui.select_vdb_tab(mock_event)
        
        # Verify event value is returned as string
        self.assertEqual(result, "Milvus")
        # Verify current_vdb_type was updated
        self.assertEqual(embedding_ui.current_vdb_type, "Milvus")


if __name__ == '__main__':
    unittest.main()