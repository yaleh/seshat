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


class TestBatchUILLMConfiguration(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        self.mock_config.llm.default_model_service = 'OpenAI'
        self.mock_config.llm.llm_services = {
            'OpenAI': MagicMock(),
            'Azure_OpenAI': MagicMock()
        }
        self.mock_config.llm.llm_services['OpenAI'].type = 'OpenAI'
        self.mock_config.llm.llm_services['OpenAI'].default_model = 'gpt-3.5-turbo'
        # Setup model args with both dict() method and keys() method
        mock_args = MagicMock()
        mock_args.dict.return_value = {
            'openai_api_key': 'test-key',
            'temperature': 0.7,
            'max_tokens': 1000,
            'request_timeout': 30,
            'max_retries': 3
        }
        mock_args.keys.return_value = ['openai_api_key', 'temperature', 'max_tokens', 'request_timeout', 'max_retries']
        self.mock_config.llm.llm_services['OpenAI'].args = mock_args
        self.mock_config.llm.llm_services['OpenAI'].models = ['gpt-3.5-turbo', 'gpt-4']
        
        self.mock_config.llm.llm_services['Azure_OpenAI'].type = 'Azure_OpenAI'
        self.mock_config.llm.llm_services['Azure_OpenAI'].default_model = 'gpt-35-turbo'
        self.mock_config.llm.llm_services['Azure_OpenAI'].args = MagicMock()
        self.mock_config.llm.llm_services['Azure_OpenAI'].args.dict.return_value = {
            'openai_api_key': 'azure-key',
            'openai_api_base': 'https://test.openai.azure.com/'
        }
        self.mock_config.llm.llm_services['Azure_OpenAI'].models = ['gpt-35-turbo', 'gpt-4']
        
        self.mock_config.server.message_db = 'test.db'
        self.mock_config.server.max_message_length = 1000
    
    @patch('ui.batch_ui.BatchUI.init_ui')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    def test_update_llm_config(self, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test update_llm_config method"""
        mock_init_ui.return_value = MagicMock()
        
        batch_ui = BatchUI(self.mock_config)
        
        # Test updating to Azure OpenAI
        batch_ui.update_llm_config('Azure_OpenAI', 'gpt-35-turbo')
        
        # Verify configuration was updated
        self.assertEqual(batch_ui.model_type, 'Azure_OpenAI')
        self.assertEqual(batch_ui.model_name, 'gpt-35-turbo')
        expected_args = {
            'openai_api_key': 'azure-key',
            'openai_api_base': 'https://test.openai.azure.com/'
        }
        self.assertEqual(batch_ui.model_args.dict(), expected_args)
    
    @patch('ui.batch_ui.BatchUI.init_ui')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    def test_show_other_llm_option_with_temperature(self, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test show_other_llm_option with temperature support"""
        mock_init_ui.return_value = MagicMock()
        
        batch_ui = BatchUI(self.mock_config)
        
        with patch('ui.batch_ui.gr.Slider') as mock_slider:
            mock_slider_instances = [MagicMock() for _ in range(4)]
            mock_slider.side_effect = mock_slider_instances
            
            result = batch_ui.show_other_llm_option(True)
            
            # Verify 4 sliders were created (temperature, max_tokens, timeout, retries)
            self.assertEqual(mock_slider.call_count, 4)
            self.assertEqual(len(result), 4)
            
            # Verify temperature slider is visible
            temp_call = mock_slider.call_args_list[0]
            self.assertTrue(temp_call.kwargs['visible'])
            self.assertEqual(temp_call.kwargs['label'], "LLM Model Temperature")
    
    @patch('ui.batch_ui.BatchUI.init_ui')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    def test_show_other_llm_option_without_temperature(self, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test show_other_llm_option without temperature support"""
        mock_init_ui.return_value = MagicMock()
        
        # Remove temperature from model args
        mock_args_no_temp = MagicMock()
        mock_args_no_temp.dict.return_value = {'openai_api_key': 'test-key'}
        mock_args_no_temp.keys.return_value = ['openai_api_key']
        self.mock_config.llm.llm_services['OpenAI'].args = mock_args_no_temp
        
        batch_ui = BatchUI(self.mock_config)
        
        with patch('ui.batch_ui.gr.Slider') as mock_slider:
            mock_slider_instances = [MagicMock() for _ in range(4)]
            mock_slider.side_effect = mock_slider_instances
            
            result = batch_ui.show_other_llm_option(True)
            
            # Verify 4 sliders were created
            self.assertEqual(mock_slider.call_count, 4)
            
            # Verify temperature slider is not visible
            temp_call = mock_slider.call_args_list[0]
            self.assertFalse(temp_call.kwargs['visible'])
    
    @patch('ui.batch_ui.BatchUI.init_ui')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    def test_update_llm_model_name_dropdown(self, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test update_llm_model_name_dropdown method"""
        mock_init_ui.return_value = MagicMock()
        
        batch_ui = BatchUI(self.mock_config)
        
        with patch('ui.batch_ui.gr.Dropdown') as mock_dropdown:
            mock_dropdown_instance = MagicMock()
            mock_dropdown.return_value = mock_dropdown_instance
            
            result = batch_ui.update_llm_model_name_dropdown('Azure_OpenAI')
            
            # Verify dropdown was created with correct models
            mock_dropdown.assert_called_once_with(choices=['gpt-35-turbo', 'gpt-4'])
            self.assertEqual(batch_ui.default_model_service, 'Azure_OpenAI')
            self.assertEqual(result, mock_dropdown_instance)


class TestBatchUIMessageHandling(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        self.mock_config.llm.default_model_service = 'OpenAI'
        self.mock_config.llm.llm_services = {'OpenAI': MagicMock()}
        self.mock_config.llm.llm_services['OpenAI'].type = 'OpenAI'
        self.mock_config.llm.llm_services['OpenAI'].default_model = 'gpt-3.5-turbo'
        self.mock_config.llm.llm_services['OpenAI'].args = MagicMock()
        self.mock_config.llm.llm_services['OpenAI'].args.dict.return_value = {'openai_api_key': 'test-key'}
        self.mock_config.server.message_db = 'test.db'
        self.mock_config.server.max_message_length = 1000
    
    @patch('ui.batch_ui.BatchUI.init_ui')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    def test_update_system_msg_and_user_msg(self, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test update_system_msg_and_user_msg method"""
        mock_init_ui.return_value = MagicMock()
        mock_db_instance = MagicMock()
        mock_db_manager.return_value = mock_db_instance
        
        # Setup database return values
        mock_db_instance.get_messages.side_effect = [
            ['system msg 1', 'system msg 2'],  # system_messages
            ['user msg 1', 'user msg 2']       # user_messages
        ]
        
        batch_ui = BatchUI(self.mock_config)
        
        with patch('ui.batch_ui.gr.update') as mock_gr_update:
            mock_gr_update.side_effect = ['system_update', 'user_update']
            
            system_result, user_result = batch_ui.update_system_msg_and_user_msg()
            
            # Verify database calls
            self.assertEqual(mock_db_instance.get_messages.call_count, 2)
            mock_db_instance.get_messages.assert_any_call("system_messages")
            mock_db_instance.get_messages.assert_any_call("user_messages")
            
            # Verify gr.update calls
            self.assertEqual(mock_gr_update.call_count, 2)
            self.assertEqual(system_result, 'system_update')
            self.assertEqual(user_result, 'user_update')
    
    @patch('ui.batch_ui.BatchUI.init_ui')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    def test_update_dropdown_choices_with_matching_string(self, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test update_dropdown_choices with matching string"""
        mock_init_ui.return_value = MagicMock()
        mock_db_instance = MagicMock()
        mock_db_manager.return_value = mock_db_instance
        
        # Setup database return values
        mock_db_instance.get_messages.return_value = [
            'hello world',
            'goodbye world',
            'hello universe',
            'test message'
        ]
        
        batch_ui = BatchUI(self.mock_config)
        
        with patch('ui.batch_ui.gr.update') as mock_gr_update:
            mock_gr_update.return_value = 'filtered_update'
            
            result = batch_ui.update_dropdown_choices('test_table', 'hello')
            
            # Verify database call
            mock_db_instance.get_messages.assert_called_once_with('test_table')
            
            # Verify gr.update was called with filtered choices
            mock_gr_update.assert_called_once_with(choices=['hello world', 'hello universe'])
            self.assertEqual(result, 'filtered_update')
    
    @patch('ui.batch_ui.BatchUI.init_ui')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    def test_update_dropdown_choices_with_empty_string(self, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test update_dropdown_choices with empty string"""
        mock_init_ui.return_value = MagicMock()
        mock_db_instance = MagicMock()
        mock_db_manager.return_value = mock_db_instance
        
        mock_db_instance.get_messages.return_value = ['msg1', 'msg2', 'msg3']
        
        batch_ui = BatchUI(self.mock_config)
        
        with patch('ui.batch_ui.gr.update') as mock_gr_update:
            mock_gr_update.return_value = 'all_messages_update'
            
            result = batch_ui.update_dropdown_choices('test_table', '')
            
            # Verify gr.update was called with all messages
            mock_gr_update.assert_called_once_with(choices=['msg1', 'msg2', 'msg3'])
            self.assertEqual(result, 'all_messages_update')
    
    @patch('ui.batch_ui.BatchUI.init_ui')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    def test_update_dropdown_choices_exact_match(self, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test update_dropdown_choices with exact match"""
        mock_init_ui.return_value = MagicMock()
        mock_db_instance = MagicMock()
        mock_db_manager.return_value = mock_db_instance
        
        mock_db_instance.get_messages.return_value = ['exact match', 'other message']
        
        batch_ui = BatchUI(self.mock_config)
        
        with patch('ui.batch_ui.gr.update') as mock_gr_update:
            mock_gr_update.return_value = 'exact_match_update'
            
            result = batch_ui.update_dropdown_choices('test_table', 'exact match')
            
            # Should return all messages when there's an exact match
            mock_gr_update.assert_called_once_with(choices=['exact match', 'other message'])
            self.assertEqual(result, 'exact_match_update')


class TestBatchUILLMOperations(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        self.mock_config.llm.default_model_service = 'OpenAI'
        self.mock_config.llm.llm_services = {'OpenAI': MagicMock()}
        self.mock_config.llm.llm_services['OpenAI'].type = 'OpenAI'
        self.mock_config.llm.llm_services['OpenAI'].default_model = 'gpt-3.5-turbo'
        self.mock_config.llm.llm_services['OpenAI'].args = MagicMock()
        self.mock_config.llm.llm_services['OpenAI'].args.dict.return_value = {'openai_api_key': 'test-key'}
        self.mock_config.server.message_db = 'test.db'
        self.mock_config.server.max_message_length = 1000
    
    @patch('ui.batch_ui.BatchUI.init_ui')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    @patch('ui.batch_ui.SystemMessage')
    @patch('ui.batch_ui.HumanMessage')
    @patch('ui.batch_ui.StrOutputParser')
    def test_send_call_func_success(self, mock_str_parser, mock_human_message, mock_system_message, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test successful send_call_func operation"""
        mock_init_ui.return_value = MagicMock()
        mock_db_instance = MagicMock()
        mock_db_manager.return_value = mock_db_instance
        
        mock_factory_instance = MagicMock()
        mock_model_factory.return_value = mock_factory_instance
        
        # Setup LLM chain mock
        mock_llm = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "AI response"
        mock_llm.__or__ = MagicMock(return_value=mock_chain)
        mock_factory_instance.create_model.return_value = mock_llm
        
        mock_system_msg = MagicMock()
        mock_human_msg = MagicMock()
        mock_system_message.return_value = mock_system_msg
        mock_human_message.return_value = mock_human_msg
        
        batch_ui = BatchUI(self.mock_config)
        
        with patch('ui.batch_ui.gr.Info'), \
             patch('ui.batch_ui.gr.Chatbot') as mock_gr_chatbot:
            mock_gr_chatbot.return_value = "chatbot_update"
            
            chat_history = [("previous user", "previous ai")]
            result = batch_ui.send_call_func("system prompt", "user prompt", chat_history)
            
            # Verify database calls
            mock_db_instance.append_message.assert_any_call("user_messages", "user prompt")
            mock_db_instance.append_message.assert_any_call("system_messages", "system prompt")
            
            # Verify model creation
            mock_factory_instance.create_model.assert_called_once_with(
                model_type='OpenAI',
                model_name='gpt-3.5-turbo',
                openai_api_key='test-key'
            )
            
            # Verify messages were created
            mock_system_message.assert_called_once_with(content="system prompt")
            mock_human_message.assert_called_once_with(content="user prompt")
            
            # Verify chain invocation
            mock_chain.invoke.assert_called_once_with([mock_system_msg, mock_human_msg])
            
            # Verify result
            self.assertEqual(result, "chatbot_update")
    
    @patch('ui.batch_ui.BatchUI.init_ui')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    def test_send_call_func_error(self, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test send_call_func with error"""
        mock_init_ui.return_value = MagicMock()
        mock_db_instance = MagicMock()
        mock_db_manager.return_value = mock_db_instance
        
        mock_factory_instance = MagicMock()
        mock_factory_instance.create_model.side_effect = Exception("Model creation failed")
        mock_model_factory.return_value = mock_factory_instance
        
        batch_ui = BatchUI(self.mock_config)
        
        with patch('ui.batch_ui.gr.Info'), \
             patch('ui.batch_ui.gr.Error', side_effect=Exception) as mock_gr_error:
            
            with self.assertRaises(Exception):
                batch_ui.send_call_func("system prompt", "user prompt", [])
            
            # Verify gr.Error was called
            mock_gr_error.assert_called_once()
    
    @patch('ui.batch_ui.BatchUI.init_ui')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    def test_create_chat_chain(self, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test create_chat_chain method"""
        mock_init_ui.return_value = MagicMock()
        
        mock_factory_instance = MagicMock()
        mock_model_factory.return_value = mock_factory_instance
        
        mock_llm = MagicMock()
        mock_factory_instance.create_model.return_value = mock_llm
        
        batch_ui = BatchUI(self.mock_config)
        
        with patch('ui.batch_ui.StrOutputParser') as mock_str_parser:
            mock_parser_instance = MagicMock()
            mock_str_parser.return_value = mock_parser_instance
            
            mock_chat_prompt = MagicMock()
            mock_chain = MagicMock()
            mock_chat_prompt.__or__ = MagicMock(return_value=mock_chain)
            
            result = batch_ui.create_chat_chain(mock_chat_prompt)
            
            # Verify model creation
            mock_factory_instance.create_model.assert_called_once_with(
                model_type='OpenAI',
                model_name='gpt-3.5-turbo',
                openai_api_key='test-key'
            )
            
            # Verify output parser creation
            mock_str_parser.assert_called_once()
            
            # The result should be the chain created by the __or__ operation
            self.assertIsNotNone(result)
    
    @patch('ui.batch_ui.BatchUI.init_ui')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    def test_cancel_batch_send(self, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test cancel_batch_send method"""
        mock_init_ui.return_value = MagicMock()
        
        batch_ui = BatchUI(self.mock_config)
        
        # Should not raise any exception (currently just passes)
        result = batch_ui.cancel_batch_send()
        self.assertIsNone(result)


class TestBatchUITableOperations(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        self.mock_config.llm.default_model_service = 'OpenAI'
        self.mock_config.llm.llm_services = {'OpenAI': MagicMock()}
        self.mock_config.llm.llm_services['OpenAI'].type = 'OpenAI'
        self.mock_config.llm.llm_services['OpenAI'].default_model = 'gpt-3.5-turbo'
        self.mock_config.llm.llm_services['OpenAI'].args = MagicMock()
        self.mock_config.llm.llm_services['OpenAI'].args.dict.return_value = {'openai_api_key': 'test-key'}
        self.mock_config.server.message_db = 'test.db'
        self.mock_config.server.max_message_length = 1000
    
    @patch('ui.batch_ui.BatchUI.init_ui')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    @patch('ui.batch_ui.TableParser')
    def test_update_table_markdown(self, mock_table_parser, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test update_table with Markdown Table method"""
        mock_init_ui.return_value = MagicMock()
        
        # Setup table parser mock
        test_df = pd.DataFrame({'col1': ['a', 'b'], 'col2': ['c', 'd']})
        mock_table_parser.parse_markdown_table.return_value = test_df
        mock_table_parser.add_skip_column.return_value = test_df
        
        batch_ui = BatchUI(self.mock_config)
        
        chatbot = [["user", "| col1 | col2 |\n|------|------|\n| a    | c    |\n| b    | d    |"]]
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file, \
             patch('ui.batch_ui.gr.File') as mock_gr_file:
            
            mock_file = MagicMock()
            mock_file.name = '/tmp/test.csv'
            mock_temp_file.return_value.__enter__.return_value = mock_file
            
            mock_gr_file.return_value = "file_update"
            
            with patch.object(test_df, 'to_csv') as mock_to_csv:
                data_result, file_result = batch_ui.update_table(chatbot, "Markdown Table")
                
                # Verify table parsing
                mock_table_parser.parse_markdown_table.assert_called_once_with(chatbot[-1][-1])
                mock_table_parser.add_skip_column.assert_called_once_with(test_df)
                
                # Verify file operations
                mock_temp_file.assert_called_once_with(suffix=".csv", delete=False)
                mock_to_csv.assert_called_once_with('/tmp/test.csv', index=False)
                mock_gr_file.assert_called_once_with(value='/tmp/test.csv', visible=True)
                
                # Verify results
                pd.testing.assert_frame_equal(data_result, test_df)
                self.assertEqual(file_result, "file_update")
    
    @patch('ui.batch_ui.BatchUI.init_ui')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    def test_update_table_chat_history(self, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test update_table with Chat History method"""
        mock_init_ui.return_value = MagicMock()
        
        batch_ui = BatchUI(self.mock_config)
        
        chatbot = [["user1", "ai1"], ["user2", "ai2"]]
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file, \
             patch('ui.batch_ui.gr.File') as mock_gr_file, \
             patch('pandas.DataFrame') as mock_pd_dataframe:
            
            mock_file = MagicMock()
            mock_file.name = '/tmp/test.csv'
            mock_temp_file.return_value.__enter__.return_value = mock_file
            
            mock_gr_file.return_value = "file_update"
            
            test_df = MagicMock()
            mock_pd_dataframe.return_value = test_df
            
            data_result, file_result = batch_ui.update_table(chatbot, "Chat History")
            
            # Verify DataFrame creation
            mock_pd_dataframe.assert_called_once_with(chatbot, columns=["Input", "Output"])
            
            # Verify file operations
            mock_temp_file.assert_called_once_with(suffix=".csv", delete=False)
            test_df.to_csv.assert_called_once_with('/tmp/test.csv', index=False)
            mock_gr_file.assert_called_once_with(value='/tmp/test.csv', visible=True)
            
            # Verify results
            self.assertEqual(data_result, test_df)
            self.assertEqual(file_result, "file_update")
    
    @patch('ui.batch_ui.BatchUI.init_ui')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    @patch('ui.batch_ui.TableParser')
    def test_update_table_history_with_tables(self, mock_table_parser, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test update_table with History w/ Tables method"""
        mock_init_ui.return_value = MagicMock()
        
        test_df = pd.DataFrame({'col1': ['a', 'b'], 'col2': ['c', 'd']})
        mock_table_parser.parse_markdown_table_history.return_value = test_df
        
        batch_ui = BatchUI(self.mock_config)
        
        chatbot = [["user", "ai with table"]]
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file, \
             patch('ui.batch_ui.gr.File') as mock_gr_file:
            
            mock_file = MagicMock()
            mock_file.name = '/tmp/test.csv'
            mock_temp_file.return_value.__enter__.return_value = mock_file
            
            mock_gr_file.return_value = "file_update"
            
            with patch.object(test_df, 'to_csv') as mock_to_csv:
                data_result, file_result = batch_ui.update_table(chatbot, "History w/ Tables")
                
                # Verify table parsing
                mock_table_parser.parse_markdown_table_history.assert_called_once_with(chatbot)
                
                # Verify file operations
                mock_temp_file.assert_called_once_with(suffix=".csv", delete=False)
                mock_to_csv.assert_called_once_with('/tmp/test.csv', index=False)
                mock_gr_file.assert_called_once_with(value='/tmp/test.csv', visible=True)
                
                # Verify results
                pd.testing.assert_frame_equal(data_result, test_df)
                self.assertEqual(file_result, "file_update")
    
    @patch('ui.batch_ui.BatchUI.init_ui')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    def test_update_table_invalid_method(self, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test update_table with invalid method"""
        mock_init_ui.return_value = MagicMock()
        
        batch_ui = BatchUI(self.mock_config)
        
        chatbot = [["user", "ai"]]
        
        data_result, file_result = batch_ui.update_table(chatbot, "Invalid Method")
        
        # Should return None for both values
        self.assertIsNone(data_result)
        self.assertIsNone(file_result)
    
    @patch('ui.batch_ui.BatchUI.init_ui')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    @patch('ui.batch_ui.TableParser')
    def test_update_table_markdown_error(self, mock_table_parser, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test update_table with Markdown Table parsing error"""
        mock_init_ui.return_value = MagicMock()
        
        # Setup table parser to raise error
        mock_table_parser.parse_markdown_table.side_effect = IndexError("No table found")
        
        batch_ui = BatchUI(self.mock_config)
        
        chatbot = [["user", "no table here"]]
        
        data_result, file_result = batch_ui.update_table(chatbot, "Markdown Table")
        
        # Should return None for both values when there's an IndexError
        self.assertIsNone(data_result)
        self.assertIsNone(file_result)
    
    @patch('ui.batch_ui.BatchUI.init_ui')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    @patch('ui.batch_ui.detect_encoding')
    @patch('ui.batch_ui.TableParser')
    def test_upload_table_csv(self, mock_table_parser, mock_detect_encoding, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test upload_table with CSV file"""
        mock_init_ui.return_value = MagicMock()
        mock_detect_encoding.return_value = 'utf-8'
        
        test_df = pd.DataFrame({'col1': ['a', 'b'], 'col2': ['c', 'd']})
        test_df_with_skip = test_df.copy()
        test_df_with_skip['Skip'] = ''
        mock_table_parser.add_skip_column.return_value = test_df_with_skip
        
        batch_ui = BatchUI(self.mock_config)
        
        mock_file = MagicMock()
        mock_file.name = 'test.csv'
        
        with patch('pandas.read_csv') as mock_read_csv, \
             patch('ui.batch_ui.gr.update') as mock_gr_update:
            
            mock_read_csv.return_value = test_df
            mock_gr_update.side_effect = ['start_update', 'end_update', 'size_update']
            
            result_df, start_result, end_result, size_result = batch_ui.upload_table(mock_file)
            
            # Verify file reading
            mock_detect_encoding.assert_called_once_with('test.csv')
            mock_read_csv.assert_called_once_with('test.csv', encoding='utf-8')
            
            # Verify skip column addition
            mock_table_parser.add_skip_column.assert_called_once_with(test_df)
            
            # Verify gr.update calls
            self.assertEqual(mock_gr_update.call_count, 3)
            
            # Verify results
            pd.testing.assert_frame_equal(result_df, test_df_with_skip)
            self.assertEqual(start_result, 'start_update')
            self.assertEqual(end_result, 'end_update')
            self.assertEqual(size_result, 'size_update')
    
    @patch('ui.batch_ui.BatchUI.init_ui')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    @patch('ui.batch_ui.TableParser')
    def test_upload_table_excel(self, mock_table_parser, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test upload_table with Excel file"""
        mock_init_ui.return_value = MagicMock()
        
        test_df = pd.DataFrame({'col1': ['a', 'b'], 'col2': ['c', 'd']})
        test_df_with_skip = test_df.copy()
        test_df_with_skip['Skip'] = ''
        mock_table_parser.add_skip_column.return_value = test_df_with_skip
        
        batch_ui = BatchUI(self.mock_config)
        
        mock_file = MagicMock()
        mock_file.name = 'test.xlsx'
        
        with patch('pandas.read_excel') as mock_read_excel, \
             patch('ui.batch_ui.gr.update') as mock_gr_update:
            
            mock_read_excel.return_value = test_df
            mock_gr_update.side_effect = ['start_update', 'end_update', 'size_update']
            
            result_df, start_result, end_result, size_result = batch_ui.upload_table(mock_file)
            
            # Verify file reading
            mock_read_excel.assert_called_once_with('test.xlsx')
            
            # Verify skip column addition
            mock_table_parser.add_skip_column.assert_called_once_with(test_df)
            
            # Verify results
            pd.testing.assert_frame_equal(result_df, test_df_with_skip)
    
    @patch('ui.batch_ui.BatchUI.init_ui')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    @patch('ui.batch_ui.TableParser')
    def test_upload_table_unsupported_format(self, mock_table_parser, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test upload_table with unsupported file format"""
        mock_init_ui.return_value = MagicMock()
        
        empty_df = pd.DataFrame()
        empty_df_with_skip = empty_df.copy()
        mock_table_parser.add_skip_column.return_value = empty_df_with_skip
        
        batch_ui = BatchUI(self.mock_config)
        
        mock_file = MagicMock()
        mock_file.name = 'test.txt'
        
        with patch('ui.batch_ui.gr.update') as mock_gr_update:
            mock_gr_update.side_effect = ['start_update', 'end_update', 'size_update']
            
            result_df, start_result, end_result, size_result = batch_ui.upload_table(mock_file)
            
            # Should process empty DataFrame
            mock_table_parser.add_skip_column.assert_called_once()
            pd.testing.assert_frame_equal(result_df, empty_df_with_skip)
    
    @patch('ui.batch_ui.BatchUI.init_ui')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    def test_upload_table_error(self, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test upload_table with file reading error"""
        mock_init_ui.return_value = MagicMock()
        
        batch_ui = BatchUI(self.mock_config)
        
        mock_file = MagicMock()
        mock_file.name = 'test.csv'
        
        with patch('ui.batch_ui.detect_encoding', return_value='utf-8'), \
             patch('pandas.read_csv', side_effect=Exception("File read error")), \
             patch('ui.batch_ui.gr.Error', side_effect=Exception) as mock_gr_error:
            
            with self.assertRaises(Exception):
                batch_ui.upload_table(mock_file)
            
            # Verify gr.Error was called
            mock_gr_error.assert_called_once()
    
    @patch('ui.batch_ui.BatchUI.init_ui')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    def test_get_table_rows(self, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test get_table_rows method"""
        mock_init_ui.return_value = MagicMock()
        
        batch_ui = BatchUI(self.mock_config)
        
        test_df = pd.DataFrame({'col1': ['a', 'b', 'c'], 'col2': ['d', 'e', 'f']})
        
        result = batch_ui.get_table_rows(test_df)
        
        self.assertEqual(result, 3)
    
    @patch('ui.batch_ui.BatchUI.init_ui')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    def test_merge_table_success(self, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test merge_table method success"""
        mock_init_ui.return_value = MagicMock()
        
        batch_ui = BatchUI(self.mock_config)
        
        # Create test input table with Skip column
        table_input = pd.DataFrame({
            'col1': ['a', 'b', 'c', 'd'],
            'col2': ['e', 'f', 'g', 'h'],
            'Skip': ['', 'skip', '', '']
        })
        
        # Create test output table
        table_output = pd.DataFrame({
            'output': ['out1', 'out2']
        })
        
        with patch('pandas.concat') as mock_concat:
            expected_result = pd.DataFrame({
                'col1': ['a', 'c', 'd'],
                'col2': ['e', 'g', 'h'],
                'Skip': ['', '', ''],
                'output': ['out1', 'out2', None]
            })
            mock_concat.return_value = expected_result
            
            result = batch_ui.merge_table(table_input, table_output, 0, 4)
            
            # Verify pandas.concat was called
            mock_concat.assert_called_once()
            
            # Verify pandas.concat was called
            mock_concat.assert_called_once()
            
            # Since we mocked pandas.concat, just verify it returned the mocked result
            self.assertIsNotNone(result)
    
    @patch('ui.batch_ui.BatchUI.init_ui')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    def test_merge_table_error(self, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test merge_table method with error"""
        mock_init_ui.return_value = MagicMock()
        
        batch_ui = BatchUI(self.mock_config)
        
        # Create empty table that will cause error
        table_input = pd.DataFrame()
        table_output = pd.DataFrame()
        
        with patch('ui.batch_ui.gr.Error', side_effect=Exception) as mock_gr_error:
            with self.assertRaises(Exception):
                batch_ui.merge_table(table_input, table_output, 0, 1)
            
            # Verify gr.Error was called
            mock_gr_error.assert_called_once()
    
    @patch('ui.batch_ui.BatchUI.init_ui')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    def test_save_dataframe_csv(self, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test save_dataframe with CSV format"""
        mock_init_ui.return_value = MagicMock()
        
        batch_ui = BatchUI(self.mock_config)
        
        test_df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file, \
             patch('ui.batch_ui.gr.update') as mock_gr_update:
            
            mock_file = MagicMock()
            mock_file.name = '/tmp/test.csv'
            mock_temp_file.return_value.__enter__.return_value = mock_file
            
            mock_gr_update.return_value = "file_update"
            
            with patch.object(test_df, 'to_csv') as mock_to_csv:
                result = batch_ui.save_dataframe('csv', test_df)
                
                # Verify tempfile was created
                mock_temp_file.assert_called_once_with(suffix=".csv", delete=False)
                
                # Verify CSV was saved
                mock_to_csv.assert_called_once_with('/tmp/test.csv', index=False)
                
                # Verify gr.update was called
                mock_gr_update.assert_called_once_with(value='/tmp/test.csv', visible=True)
                
                self.assertEqual(result, "file_update")
    
    @patch('ui.batch_ui.BatchUI.init_ui')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    def test_save_dataframe_excel(self, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test save_dataframe with Excel format"""
        mock_init_ui.return_value = MagicMock()
        
        batch_ui = BatchUI(self.mock_config)
        
        test_df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file, \
             patch('ui.batch_ui.gr.update') as mock_gr_update:
            
            mock_file = MagicMock()
            mock_file.name = '/tmp/test.xlsx'
            mock_temp_file.return_value.__enter__.return_value = mock_file
            
            mock_gr_update.return_value = "file_update"
            
            with patch.object(test_df, 'to_excel') as mock_to_excel:
                result = batch_ui.save_dataframe('xlsx', test_df)
                
                # Verify tempfile was created
                mock_temp_file.assert_called_once_with(suffix=".xlsx", delete=False)
                
                # Verify Excel was saved
                mock_to_excel.assert_called_once_with('/tmp/test.xlsx', sheet_name="sheet1", index=False)
                
                # Verify gr.update was called
                mock_gr_update.assert_called_once_with(value='/tmp/test.xlsx', visible=True)
                
                self.assertEqual(result, "file_update")


class TestBatchUIBatchProcessing(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        self.mock_config.llm.default_model_service = 'OpenAI'
        self.mock_config.llm.llm_services = {'OpenAI': MagicMock()}
        self.mock_config.llm.llm_services['OpenAI'].type = 'OpenAI'
        self.mock_config.llm.llm_services['OpenAI'].default_model = 'gpt-3.5-turbo'
        self.mock_config.llm.llm_services['OpenAI'].args = MagicMock()
        self.mock_config.llm.llm_services['OpenAI'].args.dict.return_value = {'openai_api_key': 'test-key'}
        self.mock_config.server.message_db = 'test.db'
        self.mock_config.server.max_message_length = 1000
        self.mock_config.batch = MagicMock()
        self.mock_config.batch.stop_on_error = False
    
    @patch('ui.batch_ui.BatchUI.init_ui')
    @patch('ui.batch_ui.DatabaseManager')
    @patch('ui.batch_ui.LLMModelFactory')
    @patch('ui.batch_ui.SystemMessagePromptTemplate')
    @patch('ui.batch_ui.HumanMessagePromptTemplate')
    @patch('ui.batch_ui.HumanMessage')
    @patch('ui.batch_ui.AIMessage')
    def test_create_chat_prompt(self, mock_ai_message, mock_human_message, mock_human_template, mock_system_template, mock_model_factory, mock_db_manager, mock_init_ui):
        """Test __create_chat_prompt method"""
        mock_init_ui.return_value = MagicMock()
        mock_db_instance = MagicMock()
        mock_db_manager.return_value = mock_db_instance
        
        # Setup template mocks
        mock_system_prompt = MagicMock()
        mock_system_template.from_template.return_value = mock_system_prompt
        
        mock_human_prompt = MagicMock()
        mock_human_template.from_template.return_value = mock_human_prompt
        
        mock_human_msg = MagicMock()
        mock_ai_msg = MagicMock()
        mock_human_message.return_value = mock_human_msg
        mock_ai_message.return_value = mock_ai_msg
        
        # Setup chat prompt composition
        mock_system_prompt.__add__ = MagicMock(return_value=mock_system_prompt)
        
        batch_ui = BatchUI(self.mock_config)
        
        chat_history = [("user1", "ai1"), ("user2", "ai2")]
        
        result = batch_ui._BatchUI__create_chat_prompt("system prompt", "user prompt", chat_history)
        
        # Verify database calls
        mock_db_instance.append_message.assert_any_call("user_messages", "user prompt")
        mock_db_instance.append_message.assert_any_call("system_messages", "system prompt")
        
        # Verify template creation
        mock_system_template.from_template.assert_called_once_with("system prompt")
        mock_human_template.from_template.assert_called_once_with("user prompt")
        
        # Verify human and AI messages were created for chat history
        self.assertEqual(mock_human_message.call_count, 2)
        self.assertEqual(mock_ai_message.call_count, 2)
        
        # Verify the final prompt was returned (it will be the result of += operations)
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()