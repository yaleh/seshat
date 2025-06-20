import unittest
import pandas as pd
import tempfile
from unittest.mock import patch, MagicMock, Mock
import gradio as gr

from ui.langserve_client_ui import LangServeClientUI, _dataframe_to_batch_message, _dump_dataframe


class TestLangServeClientUIBasics(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        self.mock_config.llm.default_model_service = 'OpenAI'
        self.mock_config.llm.llm_services = {
            'OpenAI': MagicMock()
        }
        self.mock_config.llm.llm_services['OpenAI'].type = 'OpenAI'
        self.mock_config.llm.llm_services['OpenAI'].default_model = 'gpt-3.5-turbo'
        self.mock_config.llm.llm_services['OpenAI'].args = {'openai_api_key': 'test-key'}
        
        self.mock_config.server.message_db = 'test.db'
        self.mock_config.server.max_message_length = 1000
    
    @patch('ui.langserve_client_ui.LangServeClientUI.bind_events')
    @patch('ui.langserve_client_ui.DatabaseManager')
    @patch('ui.langserve_client_ui.LLMModelFactory')
    def test_langserve_client_ui_init_basics(self, mock_model_factory, mock_db_manager, mock_bind_events):
        """Test LangServeClientUI basic initialization without UI creation"""
        # Setup mocks
        mock_db_instance = MagicMock()
        mock_db_manager.return_value = mock_db_instance
        mock_db_instance.get_messages.return_value = []
        
        mock_factory_instance = MagicMock()
        mock_model_factory.return_value = mock_factory_instance
        
        with patch('ui.langserve_client_ui.gr.Blocks') as mock_blocks:
            mock_ui = MagicMock()
            mock_blocks.return_value.__enter__.return_value = mock_ui
            
            with patch('ui.langserve_client_ui.gr.Tab'), \
                 patch('ui.langserve_client_ui.gr.Row'), \
                 patch('ui.langserve_client_ui.gr.Column'), \
                 patch('ui.langserve_client_ui.gr.Group'), \
                 patch('ui.langserve_client_ui.gr.Textbox'), \
                 patch('ui.langserve_client_ui.gr.Dropdown'), \
                 patch('ui.langserve_client_ui.gr.Button'), \
                 patch('ui.langserve_client_ui.gr.Dataframe'), \
                 patch('ui.langserve_client_ui.gr.File'), \
                 patch('ui.langserve_client_ui.gr.Number'), \
                 patch('ui.langserve_client_ui.gr.ClearButton'):
                
                # Create LangServeClientUI
                client_ui = LangServeClientUI(self.mock_config)
                
                # Verify initialization
                self.assertEqual(client_ui.config, self.mock_config)
                self.assertEqual(client_ui.default_model_service, 'OpenAI')
                self.assertEqual(client_ui.model_type, 'OpenAI')
                self.assertEqual(client_ui.model_name, 'gpt-3.5-turbo')
                self.assertEqual(client_ui.model_args, {'openai_api_key': 'test-key'})
                
                # Verify DatabaseManager was created correctly
                mock_db_manager.assert_called_once_with('test.db', 1000)
                self.assertEqual(client_ui.db_manager, mock_db_instance)
                
                # Verify LLMModelFactory was created
                mock_model_factory.assert_called_once()
                self.assertEqual(client_ui.model_factory, mock_factory_instance)
                
                # Verify UI was created
                self.assertEqual(client_ui.ui, mock_ui)
    
    @patch('ui.langserve_client_ui.DatabaseManager')
    @patch('ui.langserve_client_ui.LLMModelFactory')
    def test_langserve_client_ui_init_with_none_config(self, mock_model_factory, mock_db_manager):
        """Test LangServeClientUI initialization with None config raises AttributeError"""
        with self.assertRaises(AttributeError):
            LangServeClientUI(None)


class TestLangServeClientUIInvoke(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        self.mock_config.llm.default_model_service = 'OpenAI'
        self.mock_config.llm.llm_services = {'OpenAI': MagicMock()}
        self.mock_config.llm.llm_services['OpenAI'].type = 'OpenAI'
        self.mock_config.llm.llm_services['OpenAI'].default_model = 'gpt-3.5-turbo'
        self.mock_config.llm.llm_services['OpenAI'].args = {'openai_api_key': 'test-key'}
        self.mock_config.server.message_db = 'test.db'
        self.mock_config.server.max_message_length = 1000
    
    @patch('ui.langserve_client_ui.LangServeClientUI.init_ui')
    @patch('ui.langserve_client_ui.DatabaseManager')
    @patch('ui.langserve_client_ui.LLMModelFactory')
    @patch('ui.langserve_client_ui.RemoteRunnable')
    def test_invoke_success(self, mock_remote_runnable, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test successful invoke operation"""
        # Setup mocks
        mock_db_instance = MagicMock()
        mock_db_manager.return_value = mock_db_instance
        mock_init_ui.return_value = MagicMock()
        
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "test response"
        mock_remote_runnable.return_value = mock_chain
        
        # Create LangServeClientUI
        client_ui = LangServeClientUI(self.mock_config)
        
        # Test invoke
        message = "{'input': 'test message'}"
        url = "http://localhost:8000/test"
        
        with patch('ui.langserve_client_ui.gr.update') as mock_gr_update:
            mock_gr_update.return_value = "updated_value"
            result = client_ui.invoke(message, url)
            
            # Verify database calls
            mock_db_instance.append_message.assert_any_call('langserve_messages', message)
            mock_db_instance.append_message.assert_any_call('langserve_urls', url)
            
            # Verify RemoteRunnable was called
            mock_remote_runnable.assert_called_once_with(url)
            mock_chain.invoke.assert_called_once_with({'input': 'test message'})
            
            # Verify gr.update was called with the output
            mock_gr_update.assert_called_once_with(value="test response")
            self.assertEqual(result, "updated_value")
    
    @patch('ui.langserve_client_ui.LangServeClientUI.init_ui')
    @patch('ui.langserve_client_ui.DatabaseManager')
    @patch('ui.langserve_client_ui.LLMModelFactory')
    @patch('ui.langserve_client_ui.RemoteRunnable')
    def test_invoke_error(self, mock_remote_runnable, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test invoke operation with error"""
        # Setup mocks
        mock_db_instance = MagicMock()
        mock_db_manager.return_value = mock_db_instance
        mock_init_ui.return_value = MagicMock()
        
        mock_remote_runnable.side_effect = Exception("Connection failed")
        
        # Create LangServeClientUI
        client_ui = LangServeClientUI(self.mock_config)
        
        # Test invoke with exception
        message = "{'input': 'test message'}"
        url = "http://localhost:8000/test"
        
        with patch('ui.langserve_client_ui.gr.Error', side_effect=Exception) as mock_gr_error:
            with self.assertRaises(Exception):
                client_ui.invoke(message, url)
            
            # Verify gr.Error was called
            mock_gr_error.assert_called_once()
    
    @patch('ui.langserve_client_ui.LangServeClientUI.init_ui')
    @patch('ui.langserve_client_ui.DatabaseManager')
    @patch('ui.langserve_client_ui.LLMModelFactory')
    def test_update_histories(self, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test update_histories method"""
        # Setup mocks
        mock_db_instance = MagicMock()
        mock_db_instance.get_messages.side_effect = [
            ['message1', 'message2'],  # For LANGSERVE_MESSAGES_TABLE
            ['url1', 'url2']           # For LANGSERVE_URLS_TABLE
        ]
        mock_db_manager.return_value = mock_db_instance
        mock_init_ui.return_value = MagicMock()
        
        # Create LangServeClientUI
        client_ui = LangServeClientUI(self.mock_config)
        
        with patch('ui.langserve_client_ui.gr.update') as mock_gr_update:
            mock_gr_update.side_effect = ['messages_update', 'urls_update']
            
            messages, urls = client_ui.update_histories()
            
            # Verify database calls
            self.assertEqual(mock_db_instance.get_messages.call_count, 2)
            
            # Verify gr.update calls
            self.assertEqual(mock_gr_update.call_count, 2)
            self.assertEqual(messages, 'messages_update')
            self.assertEqual(urls, 'urls_update')


class TestLangServeClientUIDataProcessing(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        self.mock_config.llm.default_model_service = 'OpenAI'
        self.mock_config.llm.llm_services = {'OpenAI': MagicMock()}
        self.mock_config.llm.llm_services['OpenAI'].type = 'OpenAI'
        self.mock_config.llm.llm_services['OpenAI'].default_model = 'gpt-3.5-turbo'
        self.mock_config.llm.llm_services['OpenAI'].args = {'openai_api_key': 'test-key'}
        self.mock_config.server.message_db = 'test.db'
        self.mock_config.server.max_message_length = 1000
    
    @patch('ui.langserve_client_ui.LangServeClientUI.init_ui')
    @patch('ui.langserve_client_ui.DatabaseManager')
    @patch('ui.langserve_client_ui.LLMModelFactory')
    def test_upload_table_csv(self, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test upload_table method with CSV file"""
        # Setup mocks
        mock_db_manager.return_value = MagicMock()
        mock_init_ui.return_value = MagicMock()
        
        # Create LangServeClientUI
        client_ui = LangServeClientUI(self.mock_config)
        
        # Create a mock file object
        mock_file = MagicMock()
        mock_file.name = 'test.csv'
        
        with patch('ui.langserve_client_ui.detect_encoding', return_value='utf-8'), \
             patch('pandas.read_csv') as mock_read_csv:
            
            test_df = pd.DataFrame({'input': ['hello', 'world']})
            mock_read_csv.return_value = test_df
            
            result = client_ui.upload_table(mock_file)
            
            # Verify file was read correctly
            mock_read_csv.assert_called_once_with('test.csv', encoding='utf-8')
            pd.testing.assert_frame_equal(result, test_df)
    
    @patch('ui.langserve_client_ui.LangServeClientUI.init_ui')
    @patch('ui.langserve_client_ui.DatabaseManager')
    @patch('ui.langserve_client_ui.LLMModelFactory')
    def test_upload_table_excel(self, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test upload_table method with Excel file"""
        # Setup mocks
        mock_db_manager.return_value = MagicMock()
        mock_init_ui.return_value = MagicMock()
        
        # Create LangServeClientUI
        client_ui = LangServeClientUI(self.mock_config)
        
        # Create a mock file object
        mock_file = MagicMock()
        mock_file.name = 'test.xlsx'
        
        with patch('pandas.read_excel') as mock_read_excel:
            test_df = pd.DataFrame({'input': ['hello', 'world']})
            mock_read_excel.return_value = test_df
            
            result = client_ui.upload_table(mock_file)
            
            # Verify file was read correctly
            mock_read_excel.assert_called_once_with('test.xlsx')
            pd.testing.assert_frame_equal(result, test_df)
    
    @patch('ui.langserve_client_ui.LangServeClientUI.init_ui')
    @patch('ui.langserve_client_ui.DatabaseManager')
    @patch('ui.langserve_client_ui.LLMModelFactory')
    def test_upload_table_unsupported_format(self, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test upload_table method with unsupported file format"""
        # Setup mocks
        mock_db_manager.return_value = MagicMock()
        mock_init_ui.return_value = MagicMock()
        
        # Create LangServeClientUI
        client_ui = LangServeClientUI(self.mock_config)
        
        # Create a mock file object with unsupported format
        mock_file = MagicMock()
        mock_file.name = 'test.txt'
        
        result = client_ui.upload_table(mock_file)
        
        # Verify empty DataFrame is returned for unsupported format
        self.assertTrue(result.empty)
    
    @patch('ui.langserve_client_ui.LangServeClientUI.init_ui')
    @patch('ui.langserve_client_ui.DatabaseManager')
    @patch('ui.langserve_client_ui.LLMModelFactory')
    def test_upload_table_error(self, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test upload_table method with file reading error"""
        # Setup mocks
        mock_db_manager.return_value = MagicMock()
        mock_init_ui.return_value = MagicMock()
        
        # Create LangServeClientUI
        client_ui = LangServeClientUI(self.mock_config)
        
        # Create a mock file object
        mock_file = MagicMock()
        mock_file.name = 'test.csv'
        
        with patch('ui.langserve_client_ui.detect_encoding', return_value='utf-8'), \
             patch('pandas.read_csv', side_effect=Exception("File read error")):
            
            with patch('ui.langserve_client_ui.gr.Error', side_effect=Exception) as mock_gr_error:
                with self.assertRaises(Exception):
                    client_ui.upload_table(mock_file)
                
                # Verify gr.Error was called
                mock_gr_error.assert_called_once()
    
    @patch('ui.langserve_client_ui.LangServeClientUI.init_ui')
    @patch('ui.langserve_client_ui.DatabaseManager')
    @patch('ui.langserve_client_ui.LLMModelFactory')
    def test_update_input_rows(self, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test update_input_rows method"""
        # Setup mocks
        mock_db_manager.return_value = MagicMock()
        mock_init_ui.return_value = MagicMock()
        
        # Create LangServeClientUI
        client_ui = LangServeClientUI(self.mock_config)
        
        # Create test dataframe
        test_df = pd.DataFrame({'input': ['msg1', 'msg2', 'msg3']})
        
        with patch('ui.langserve_client_ui.gr.update') as mock_gr_update:
            mock_gr_update.side_effect = ['rows_update', 'start_update', 'end_update', 'size_update']
            
            rows, start, end, size = client_ui.update_input_rows(test_df)
            
            # Verify gr.update calls
            self.assertEqual(mock_gr_update.call_count, 4)
            self.assertEqual(rows, 'rows_update')
            self.assertEqual(start, 'start_update')
            self.assertEqual(end, 'end_update')
            self.assertEqual(size, 'size_update')


class TestLangServeClientUIBatch(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        self.mock_config.llm.default_model_service = 'OpenAI'
        self.mock_config.llm.llm_services = {'OpenAI': MagicMock()}
        self.mock_config.llm.llm_services['OpenAI'].type = 'OpenAI'
        self.mock_config.llm.llm_services['OpenAI'].default_model = 'gpt-3.5-turbo'
        self.mock_config.llm.llm_services['OpenAI'].args = {'openai_api_key': 'test-key'}
        self.mock_config.server.message_db = 'test.db'
        self.mock_config.server.max_message_length = 1000
    
    @patch('ui.langserve_client_ui.LangServeClientUI.init_ui')
    @patch('ui.langserve_client_ui.DatabaseManager')
    @patch('ui.langserve_client_ui.LLMModelFactory')
    @patch('ui.langserve_client_ui.RemoteRunnable')
    @patch('ui.langserve_client_ui._dataframe_to_batch_message')
    def test_batch_success(self, mock_dataframe_to_batch, mock_remote_runnable, 
                          mock_model_factory, mock_db_manager, mock_init_ui):
        """Test successful batch operation"""
        # Setup mocks
        mock_db_instance = MagicMock()
        mock_db_manager.return_value = mock_db_instance
        mock_init_ui.return_value = MagicMock()
        
        mock_chain = MagicMock()
        mock_chain.batch.return_value = ["response1", "response2"]
        mock_remote_runnable.return_value = mock_chain
        
        mock_dataframe_to_batch.return_value = "[{'input': 'msg1'}, {'input': 'msg2'}]"
        
        # Create LangServeClientUI
        client_ui = LangServeClientUI(self.mock_config)
        
        # Create test dataframe
        test_df = pd.DataFrame({'input': ['msg1', 'msg2']})
        
        # Create mock progress
        mock_progress = MagicMock()
        
        with patch('ui.langserve_client_ui.gr.Info'), \
             patch('ui.langserve_client_ui.gr.update') as mock_gr_update, \
             patch('pandas.DataFrame') as mock_pd_dataframe, \
             patch('pandas.concat') as mock_pd_concat:
            
            mock_gr_update.return_value = "updated_dataframe"
            mock_result_df = MagicMock()
            mock_pd_concat.return_value = mock_result_df
            
            result = client_ui.batch(test_df, 0, 2, 2, "http://localhost:8000/test", mock_progress)
            
            # Verify database call
            mock_db_instance.append_message.assert_called_once_with('langserve_urls', "http://localhost:8000/test")
            
            # Verify RemoteRunnable was called
            mock_remote_runnable.assert_called_once_with("http://localhost:8000/test")
            mock_chain.batch.assert_called_once()
            
            # Verify gr.update was called
            mock_gr_update.assert_called_once_with(value=mock_result_df)
            self.assertEqual(result, "updated_dataframe")


class TestUtilityFunctions(unittest.TestCase):
    
    def test_dataframe_to_batch_message_basic(self):
        """Test _dataframe_to_batch_message with basic dataframe"""
        df = pd.DataFrame({
            'input': ['hello', 'world'],
            'model': ['gpt-3.5-turbo', 'gpt-4']
        })
        
        result = _dataframe_to_batch_message(df)
        
        # Result should be a formatted string representation
        self.assertIsInstance(result, str)
        self.assertIn('hello', result)
        self.assertIn('world', result)
        self.assertIn('gpt-3.5-turbo', result)
        self.assertIn('gpt-4', result)
    
    def test_dataframe_to_batch_message_with_eval_columns(self):
        """Test _dataframe_to_batch_message with eval columns (prefixed with !)"""
        df = pd.DataFrame({
            'input': ['hello', 'world'],
            '!config': ['{"temp": 0.5}', '{"temp": 0.7}']
        })
        
        result = _dataframe_to_batch_message(df)
        
        # Result should be a formatted string representation with eval applied
        self.assertIsInstance(result, str)
        self.assertIn('hello', result)
        self.assertIn('world', result)
        # The ! prefix should be removed and content should be evaluated
        self.assertIn('config', result)
        self.assertNotIn('!config', result)
    
    def test_dump_dataframe_csv(self):
        """Test _dump_dataframe with CSV format"""
        df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
            mock_file = MagicMock()
            mock_file.name = '/tmp/test.csv'
            mock_temp_file.return_value.__enter__.return_value = mock_file
            
            with patch.object(df, 'to_csv') as mock_to_csv:
                result = _dump_dataframe(df, ['csv'])
                
                # Verify tempfile was created
                mock_temp_file.assert_called_once_with(suffix=".csv", delete=False)
                
                # Verify CSV was saved with the temp file name
                mock_to_csv.assert_called_once_with('/tmp/test.csv', index=False)
                
                # Verify filename is returned
                self.assertEqual(result, ['/tmp/test.csv'])
    
    def test_dump_dataframe_xlsx(self):
        """Test _dump_dataframe with XLSX format"""
        df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
            mock_file = MagicMock()
            mock_file.name = '/tmp/test.xlsx'
            mock_temp_file.return_value.__enter__.return_value = mock_file
            
            with patch.object(df, 'to_excel') as mock_to_excel:
                result = _dump_dataframe(df, ['xlsx'])
                
                # Verify tempfile was created
                mock_temp_file.assert_called_once_with(suffix=".xlsx", delete=False)
                
                # Verify Excel was saved
                mock_to_excel.assert_called_once_with('/tmp/test.xlsx', index=False)
                
                # Verify filename is returned
                self.assertEqual(result, ['/tmp/test.xlsx'])
    
    def test_dump_dataframe_multiple_formats(self):
        """Test _dump_dataframe with multiple formats"""
        df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
            mock_files = [MagicMock(), MagicMock()]
            mock_files[0].__enter__.return_value.name = '/tmp/test.csv'
            mock_files[1].__enter__.return_value.name = '/tmp/test.xlsx'
            mock_temp_file.side_effect = mock_files
            
            with patch.object(df, 'to_csv') as mock_to_csv, \
                 patch.object(df, 'to_excel') as mock_to_excel:
                
                result = _dump_dataframe(df, ['csv', 'xlsx'])
                
                # Verify both tempfiles were created
                self.assertEqual(mock_temp_file.call_count, 2)
                
                # Verify both formats were saved
                mock_to_csv.assert_called_once_with('/tmp/test.csv', index=False)
                mock_to_excel.assert_called_once_with('/tmp/test.xlsx', index=False)
                
                # Verify both filenames are returned
                self.assertEqual(result, ['/tmp/test.csv', '/tmp/test.xlsx'])


if __name__ == '__main__':
    unittest.main()