import unittest
import pandas as pd
import tempfile
from unittest.mock import patch, MagicMock, Mock
import gradio as gr

from ui.batch_ui import BatchUI


class TestBatchUIBasics(unittest.TestCase):
    
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
    
    @patch('ui.batch_ui.BatchUI.init_ui')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    def test_batch_ui_init_basics(self, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test BatchUI basic initialization without UI creation"""
        # Setup mocks
        mock_db_instance = MagicMock()
        mock_db_manager.return_value = mock_db_instance
        
        mock_factory_instance = MagicMock()
        mock_model_factory.return_value = mock_factory_instance
        
        mock_ui_instance = MagicMock()
        mock_init_ui.return_value = mock_ui_instance
        
        # Create BatchUI
        batch_ui = BatchUI(self.mock_config)
        
        # Verify initialization
        self.assertEqual(batch_ui.config, self.mock_config)
        self.assertEqual(batch_ui.default_model_service, 'OpenAI')
        self.assertEqual(batch_ui.model_type, 'OpenAI')
        self.assertEqual(batch_ui.model_name, 'gpt-3.5-turbo')
        self.assertEqual(batch_ui.model_args, {'openai_api_key': 'test-key'})
        
        # Verify DatabaseManager was created correctly
        mock_db_manager.assert_called_once_with('test.db', 1000)
        self.assertEqual(batch_ui.db_manager, mock_db_instance)
        
        # Verify LLMModelFactory was created
        mock_model_factory.assert_called_once()
        self.assertEqual(batch_ui.model_factory, mock_factory_instance)
        
        # Verify UI was created
        self.assertEqual(batch_ui.ui, mock_ui_instance)
    
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    def test_batch_ui_init_with_none_config(self, mock_model_factory, mock_db_manager):
        """Test BatchUI initialization with None config raises AttributeError"""
        with self.assertRaises(AttributeError):
            BatchUI(None)
    
    @patch('ui.batch_ui.BatchUI.bind_events')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    @patch('ui.batch_ui.gr.Blocks')
    def test_batch_ui_init_stores_components(self, mock_blocks, mock_model_factory, mock_db_manager, mock_bind_events):
        """Test that BatchUI properly stores UI components"""
        mock_ui = MagicMock()
        mock_blocks.return_value.__enter__.return_value = mock_ui
        
        with patch('ui.batch_ui.gr.Chatbot') as mock_chatbot, \
             patch('ui.batch_ui.gr.Dropdown') as mock_dropdown, \
             patch('ui.batch_ui.gr.Button') as mock_button, \
             patch('ui.batch_ui.gr.Dataframe') as mock_dataframe, \
             patch('ui.batch_ui.gr.Row'), \
             patch('ui.batch_ui.gr.Column'), \
             patch('ui.batch_ui.gr.Tab'), \
             patch('ui.batch_ui.gr.Group'):
            
            mock_chatbot_instance = MagicMock()
            mock_chatbot.return_value = mock_chatbot_instance
            
            mock_dropdown_instance = MagicMock()
            mock_dropdown.return_value = mock_dropdown_instance
            
            mock_button_instance = MagicMock()
            mock_button.return_value = mock_button_instance
            
            mock_dataframe_instance = MagicMock()
            mock_dataframe.return_value = mock_dataframe_instance
            
            batch_ui = BatchUI(self.mock_config)
            
            # Verify components are stored
            self.assertTrue(hasattr(batch_ui, 'chatbot'))
            self.assertTrue(hasattr(batch_ui, 'updating_table_method'))
            self.assertTrue(hasattr(batch_ui, 'update_table_button'))
            self.assertTrue(hasattr(batch_ui, 'table_dataframe_output'))


class TestBatchUIConfiguration(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        self.mock_config.llm.default_model_service = 'Azure_OpenAI'
        self.mock_config.llm.llm_services = {
            'Azure_OpenAI': MagicMock()
        }
        self.mock_config.llm.llm_services['Azure_OpenAI'].type = 'Azure_OpenAI'
        self.mock_config.llm.llm_services['Azure_OpenAI'].default_model = 'gpt-35-turbo'
        self.mock_config.llm.llm_services['Azure_OpenAI'].args = {
            'openai_api_key': 'azure-key',
            'openai_api_base': 'https://test.openai.azure.com/'
        }
        
        self.mock_config.server.message_db = 'azure_test.db'
        self.mock_config.server.max_message_length = 2000
    
    @patch('ui.batch_ui.BatchUI.bind_events')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    @patch('ui.batch_ui.gr.Blocks')
    def test_batch_ui_with_azure_config(self, mock_blocks, mock_model_factory, mock_db_manager, mock_bind_events):
        """Test BatchUI with Azure OpenAI configuration"""
        mock_ui = MagicMock()
        mock_blocks.return_value.__enter__.return_value = mock_ui
        
        with patch('ui.batch_ui.gr.Chatbot'), \
             patch('ui.batch_ui.gr.Dropdown'), \
             patch('ui.batch_ui.gr.Button'), \
             patch('ui.batch_ui.gr.Dataframe'), \
             patch('ui.batch_ui.gr.Row'), \
             patch('ui.batch_ui.gr.Column'), \
             patch('ui.batch_ui.gr.Tab'), \
             patch('ui.batch_ui.gr.Group'):
            
            batch_ui = BatchUI(self.mock_config)
            
            # Verify Azure configuration is properly loaded
            self.assertEqual(batch_ui.default_model_service, 'Azure_OpenAI')
            self.assertEqual(batch_ui.model_type, 'Azure_OpenAI')
            self.assertEqual(batch_ui.model_name, 'gpt-35-turbo')
            self.assertEqual(batch_ui.model_args['openai_api_key'], 'azure-key')
            self.assertEqual(batch_ui.model_args['openai_api_base'], 'https://test.openai.azure.com/')
            
            # Verify DatabaseManager was created with Azure config
            mock_db_manager.assert_called_once_with('azure_test.db', 2000)


class TestBatchUIUIComponents(unittest.TestCase):
    
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
    
    @patch('ui.batch_ui.BatchUI.bind_events')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    @patch('ui.batch_ui.gr.Blocks')
    def test_chatbot_configuration(self, mock_blocks, mock_model_factory, mock_db_manager, mock_bind_events):
        """Test chatbot component configuration"""
        mock_ui = MagicMock()
        mock_blocks.return_value.__enter__.return_value = mock_ui
        
        with patch('ui.batch_ui.gr.Chatbot') as mock_chatbot, \
             patch('ui.batch_ui.gr.Dropdown'), \
             patch('ui.batch_ui.gr.Button'), \
             patch('ui.batch_ui.gr.Dataframe'), \
             patch('ui.batch_ui.gr.Row'), \
             patch('ui.batch_ui.gr.Column'), \
             patch('ui.batch_ui.gr.Tab'), \
             patch('ui.batch_ui.gr.Group'):
            
            batch_ui = BatchUI(self.mock_config)
            
            # Verify chatbot was created with correct parameters
            mock_chatbot.assert_called_with(show_share_button=True, show_copy_button=True)
    
    @patch('ui.batch_ui.BatchUI.bind_events')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    @patch('ui.batch_ui.gr.Blocks')
    def test_dropdown_configuration(self, mock_blocks, mock_model_factory, mock_db_manager, mock_bind_events):
        """Test dropdown component configuration"""
        mock_ui = MagicMock()
        mock_blocks.return_value.__enter__.return_value = mock_ui
        
        with patch('ui.batch_ui.gr.Chatbot'), \
             patch('ui.batch_ui.gr.Dropdown') as mock_dropdown, \
             patch('ui.batch_ui.gr.Button'), \
             patch('ui.batch_ui.gr.Dataframe'), \
             patch('ui.batch_ui.gr.Row'), \
             patch('ui.batch_ui.gr.Column'), \
             patch('ui.batch_ui.gr.Tab'), \
             patch('ui.batch_ui.gr.Group'):
            
            batch_ui = BatchUI(self.mock_config)
            
            # Verify dropdown was created with correct choices
            # Note: This will be called multiple times for different dropdowns
            dropdown_calls = mock_dropdown.call_args_list
            
            # Check if the updating_table_method dropdown was created
            updating_method_call = None
            for call in dropdown_calls:
                if 'choices' in call.kwargs and 'Markdown Table' in call.kwargs.get('choices', []):
                    updating_method_call = call
                    break
            
            self.assertIsNotNone(updating_method_call)
            expected_choices = ["Markdown Table", "Chat History", "History w/ Tables"]
            self.assertEqual(updating_method_call.kwargs['choices'], expected_choices)
            self.assertEqual(updating_method_call.kwargs['value'], "Markdown Table")
            self.assertFalse(updating_method_call.kwargs['show_label'])
            self.assertFalse(updating_method_call.kwargs['allow_custom_value'])
    
    @patch('ui.batch_ui.BatchUI.bind_events')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    @patch('ui.batch_ui.gr.Blocks')
    def test_button_creation(self, mock_blocks, mock_model_factory, mock_db_manager, mock_bind_events):
        """Test button components are created"""
        mock_ui = MagicMock()
        mock_blocks.return_value.__enter__.return_value = mock_ui
        
        with patch('ui.batch_ui.gr.Chatbot'), \
             patch('ui.batch_ui.gr.Dropdown'), \
             patch('ui.batch_ui.gr.Button') as mock_button, \
             patch('ui.batch_ui.gr.Dataframe'), \
             patch('ui.batch_ui.gr.Row'), \
             patch('ui.batch_ui.gr.Column'), \
             patch('ui.batch_ui.gr.Tab'), \
             patch('ui.batch_ui.gr.Group'):
            
            batch_ui = BatchUI(self.mock_config)
            
            # Verify buttons were created (should be called multiple times)
            self.assertGreater(mock_button.call_count, 0)
            
            # Check for specific button with "Get Table from Message" value
            button_calls = mock_button.call_args_list
            get_table_button_call = None
            for call in button_calls:
                if 'value' in call.kwargs and call.kwargs['value'] == "Get Table from Message":
                    get_table_button_call = call
                    break
            
            self.assertIsNotNone(get_table_button_call)


class TestBatchUIIntegration(unittest.TestCase):
    
    def test_batch_ui_component_integration(self):
        """Test that BatchUI integrates all components properly"""
        mock_config = MagicMock()
        mock_config.llm.default_model_service = 'OpenAI'
        mock_config.llm.llm_services = {'OpenAI': MagicMock()}
        mock_config.llm.llm_services['OpenAI'].type = 'OpenAI'
        mock_config.llm.llm_services['OpenAI'].default_model = 'gpt-3.5-turbo'
        mock_config.llm.llm_services['OpenAI'].args = {'openai_api_key': 'test-key'}
        mock_config.server.message_db = 'test.db'
        mock_config.server.max_message_length = 1000
        
        with patch('ui.batch_ui.BatchUI.bind_events'), \
             patch('ui.batch_ui.DatabaseManager') as mock_db_manager, \
             patch('ui.batch_ui.LLMModelFactory') as mock_model_factory, \
             patch('ui.batch_ui.gr.Blocks') as mock_blocks:
            
            # Setup mocks
            mock_ui = MagicMock()
            mock_blocks.return_value.__enter__.return_value = mock_ui
            mock_db_manager.return_value = MagicMock()
            mock_model_factory.return_value = MagicMock()
            
            with patch('ui.batch_ui.gr.Chatbot'), \
                 patch('ui.batch_ui.gr.Dropdown'), \
                 patch('ui.batch_ui.gr.Button'), \
                 patch('ui.batch_ui.gr.Dataframe'), \
                 patch('ui.batch_ui.gr.Row'), \
                 patch('ui.batch_ui.gr.Column'), \
                 patch('ui.batch_ui.gr.Tab'), \
                 patch('ui.batch_ui.gr.Group'):
                
                batch_ui = BatchUI(mock_config)
                
                # Verify core integration
                self.assertIsNotNone(batch_ui.config)
                self.assertIsNotNone(batch_ui.db_manager)
                self.assertIsNotNone(batch_ui.model_factory)
                self.assertIsNotNone(batch_ui.ui)
                
                # Verify configuration is properly extracted
                self.assertEqual(batch_ui.default_model_service, 'OpenAI')
                self.assertEqual(batch_ui.model_type, 'OpenAI')
                self.assertEqual(batch_ui.model_name, 'gpt-3.5-turbo')


if __name__ == '__main__':
    unittest.main()