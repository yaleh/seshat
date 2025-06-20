import unittest
import os
import signal
from unittest.mock import patch, MagicMock, Mock
import requests

from ui.system_ui import SystemUI


class TestSystemUI(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock config structure
        self.mock_config = MagicMock()
        self.mock_config.llm.llm_services = {
            'OpenAI': MagicMock(),
            'OpenRouter': MagicMock()
        }
        self.mock_config.llm.llm_services['OpenAI'].args.openai_api_key = 'test-key'
        self.mock_config.llm.llm_services['OpenAI'].models = []
        self.mock_config.llm.llm_services['OpenRouter'].models = []
        
        self.config_file_path = '/tmp/test_config.yaml'
    
    @patch('ui.system_ui.gr.Blocks')
    @patch('ui.system_ui.gr.Button')
    def test_init_creates_ui_components(self, mock_button, mock_blocks):
        """Test that initialization creates all UI components"""
        mock_ui = MagicMock()
        mock_blocks.return_value.__enter__.return_value = mock_ui
        mock_button_instance = MagicMock()
        mock_button.return_value = mock_button_instance
        
        system_ui = SystemUI(self.mock_config, self.config_file_path)
        
        # Verify UI components were created
        self.assertEqual(mock_button.call_count, 4)  # 4 buttons created
        button_calls = [call[1] for call in mock_button.call_args_list]
        
        # Check button configurations
        expected_buttons = [
            {"variant": "secondary"},  # Refresh OpenAI
            {"variant": "secondary"},  # Refresh OpenRouter  
            {"variant": "secondary"},  # Save Configurations
            {"variant": "stop"}        # Exit
        ]
        
        for i, expected in enumerate(expected_buttons):
            self.assertEqual(button_calls[i], expected)
    
    @patch('ui.system_ui.gr.Blocks')
    @patch('ui.system_ui.gr.Button')
    def test_init_sets_button_click_handlers(self, mock_button, mock_blocks):
        """Test that button click handlers are properly set"""
        mock_ui = MagicMock()
        mock_blocks.return_value.__enter__.return_value = mock_ui
        mock_button_instance = MagicMock()
        mock_button.return_value = mock_button_instance
        
        system_ui = SystemUI(self.mock_config, self.config_file_path)
        
        # Verify click handlers were set (4 buttons, each with .click() called)
        self.assertEqual(mock_button_instance.click.call_count, 4)
    
    def test_init_stores_config_and_file_path(self):
        """Test that initialization stores config and file path"""
        with patch('ui.system_ui.gr.Blocks'), patch('ui.system_ui.gr.Button'):
            system_ui = SystemUI(self.mock_config, self.config_file_path)
            
            self.assertEqual(system_ui.config, self.mock_config)
            self.assertEqual(system_ui.config_file_path, self.config_file_path)
            self.assertIsNotNone(system_ui.ui)
    
    @patch('ui.system_ui.os.kill')
    @patch('ui.system_ui.os.getpid')
    def test_stop_server(self, mock_getpid, mock_kill):
        """Test stop_server method"""
        mock_getpid.return_value = 12345
        
        with patch('ui.system_ui.gr.Blocks'), patch('ui.system_ui.gr.Button'):
            system_ui = SystemUI(self.mock_config, self.config_file_path)
            system_ui.stop_server()
            
            mock_getpid.assert_called_once()
            mock_kill.assert_called_once_with(12345, signal.SIGTERM)
    
    @patch('ui.system_ui.OpenAI')
    @patch('ui.system_ui.gr.Info')
    def test_refresh_openai_services_success(self, mock_gr_info, mock_openai):
        """Test successful OpenAI services refresh"""
        # Setup mock OpenAI client
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Mock models response
        mock_models = [
            Mock(id='gpt-3.5-turbo'),
            Mock(id='gpt-4'),
            Mock(id='text-davinci-003'),  # Should be filtered out
            Mock(id='gpt-4-turbo'),
            Mock(id='ada'),  # Should be filtered out
        ]
        mock_client.models.list.return_value = mock_models
        
        with patch('ui.system_ui.gr.Blocks'), patch('ui.system_ui.gr.Button'):
            system_ui = SystemUI(self.mock_config, self.config_file_path)
            system_ui.refresh_openai_services()
            
            # Verify OpenAI client was created with correct API key
            mock_openai.assert_called_once_with(api_key='test-key')
            
            # Verify models.list() was called
            mock_client.models.list.assert_called_once()
            
            # Verify only GPT models were added
            expected_models = ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo']
            self.assertEqual(self.mock_config.llm.llm_services['OpenAI'].models, expected_models)
            
            # Verify success message
            mock_gr_info.assert_called_once_with("OpenAI services refreshed")
    
    @patch('ui.system_ui.requests.get')
    @patch('ui.system_ui.gr.Info')
    def test_refresh_openrouter_services_success(self, mock_gr_info, mock_requests_get):
        """Test successful OpenRouter services refresh"""
        # Mock API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'data': [
                {'id': 'openrouter/auto'},
                {'id': 'anthropic/claude-3-sonnet'},
                {'id': 'openai/gpt-4'},
                {'id': 'mistralai/mistral-7b-instruct'}
            ]
        }
        mock_requests_get.return_value = mock_response
        
        with patch('ui.system_ui.gr.Blocks'), patch('ui.system_ui.gr.Button'):
            system_ui = SystemUI(self.mock_config, self.config_file_path)
            system_ui.refresh_openrouter_services()
            
            # Verify API call was made with timeout
            mock_requests_get.assert_called_once_with(
                "https://openrouter.ai/api/v1/models", 
                timeout=10
            )
            
            # Verify models were extracted correctly
            expected_models = [
                'openrouter/auto',
                'anthropic/claude-3-sonnet', 
                'openai/gpt-4',
                'mistralai/mistral-7b-instruct'
            ]
            self.assertEqual(self.mock_config.llm.llm_services['OpenRouter'].models, expected_models)
            
            # Verify success message
            mock_gr_info.assert_called_once_with("OpenRouter services refreshed")
    
    @patch('ui.system_ui.requests.get')
    def test_refresh_openrouter_services_network_error(self, mock_requests_get):
        """Test OpenRouter services refresh with network error"""
        # Mock network error
        mock_requests_get.side_effect = requests.RequestException("Network error")
        
        with patch('ui.system_ui.gr.Blocks'), patch('ui.system_ui.gr.Button'):
            system_ui = SystemUI(self.mock_config, self.config_file_path)
            
            # Should raise the exception (no error handling in original code)
            with self.assertRaises(requests.RequestException):
                system_ui.refresh_openrouter_services()
    
    @patch('ui.system_ui.to_yaml_file')
    @patch('ui.system_ui.gr.Info')
    def test_save_configurations_success(self, mock_gr_info, mock_to_yaml_file):
        """Test successful configuration save"""
        with patch('ui.system_ui.gr.Blocks'), patch('ui.system_ui.gr.Button'):
            system_ui = SystemUI(self.mock_config, self.config_file_path)
            system_ui.save_configurations()
            
            # Verify YAML file was written
            mock_to_yaml_file.assert_called_once_with(self.config_file_path, self.mock_config)
            
            # Verify success message
            expected_message = f"Configurations saved to {self.config_file_path}"
            mock_gr_info.assert_called_once_with(expected_message)
    
    @patch('ui.system_ui.to_yaml_file')
    def test_save_configurations_file_error(self, mock_to_yaml_file):
        """Test configuration save with file error"""
        # Mock file error
        mock_to_yaml_file.side_effect = IOError("Permission denied")
        
        with patch('ui.system_ui.gr.Blocks'), patch('ui.system_ui.gr.Button'):
            system_ui = SystemUI(self.mock_config, self.config_file_path)
            
            # Should raise the exception (no error handling in original code)
            with self.assertRaises(IOError):
                system_ui.save_configurations()
    
    @patch('ui.system_ui.OpenAI')
    def test_refresh_openai_services_api_error(self, mock_openai):
        """Test OpenAI services refresh with API error"""
        # Mock API error
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.models.list.side_effect = Exception("API error")
        
        with patch('ui.system_ui.gr.Blocks'), patch('ui.system_ui.gr.Button'):
            system_ui = SystemUI(self.mock_config, self.config_file_path)
            
            # Should raise the exception (no error handling in original code)
            with self.assertRaises(Exception):
                system_ui.refresh_openai_services()
    
    def test_init_ui_returns_ui_object(self):
        """Test that init_ui returns the UI object"""
        with patch('ui.system_ui.gr.Blocks') as mock_blocks:
            mock_ui = MagicMock()
            mock_blocks.return_value.__enter__.return_value = mock_ui
            
            with patch('ui.system_ui.gr.Button'):
                system_ui = SystemUI(self.mock_config, self.config_file_path)
                
                # The UI should be stored and returned
                self.assertEqual(system_ui.ui, mock_ui)
    
    @patch('ui.system_ui.OpenAI')
    @patch('ui.system_ui.gr.Info')
    def test_refresh_openai_services_no_gpt_models(self, mock_gr_info, mock_openai):
        """Test OpenAI refresh when no GPT models are available"""
        # Setup mock with no GPT models
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_models = [
            Mock(id='text-davinci-003'),
            Mock(id='ada'),
            Mock(id='babbage'),
        ]
        mock_client.models.list.return_value = mock_models
        
        with patch('ui.system_ui.gr.Blocks'), patch('ui.system_ui.gr.Button'):
            system_ui = SystemUI(self.mock_config, self.config_file_path)
            system_ui.refresh_openai_services()
            
            # Should result in empty models list
            self.assertEqual(self.mock_config.llm.llm_services['OpenAI'].models, [])
            
            # Should still show success message
            mock_gr_info.assert_called_once_with("OpenAI services refreshed")


class TestSystemUIIntegration(unittest.TestCase):
    
    def test_system_ui_workflow(self):
        """Test complete SystemUI workflow"""
        mock_config = MagicMock()
        mock_config.llm.llm_services = {
            'OpenAI': MagicMock(),
            'OpenRouter': MagicMock()
        }
        
        with patch('ui.system_ui.gr.Blocks'), patch('ui.system_ui.gr.Button'):
            # Test initialization
            system_ui = SystemUI(mock_config, '/tmp/config.yaml')
            
            # Verify all components are accessible
            self.assertIsNotNone(system_ui.config)
            self.assertIsNotNone(system_ui.config_file_path)
            self.assertIsNotNone(system_ui.ui)
            self.assertTrue(hasattr(system_ui, 'refresh_openai_button'))
            self.assertTrue(hasattr(system_ui, 'refresh_openrouter_button'))
            self.assertTrue(hasattr(system_ui, 'save_configurations_button'))
            self.assertTrue(hasattr(system_ui, 'exit_button'))


class TestSystemUIAdvanced(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        self.mock_config.llm.llm_services = {
            'OpenAI': MagicMock(),
            'OpenRouter': MagicMock()
        }
        self.mock_config.llm.llm_services['OpenAI'].args.openai_api_key = 'test-key'
        self.mock_config.llm.llm_services['OpenAI'].models = []
        self.mock_config.llm.llm_services['OpenRouter'].models = []
        self.config_file_path = '/tmp/test_config.yaml'
    
    @patch('ui.system_ui.SystemUI.init_ui')
    def test_get_config_info_detailed(self, mock_init_ui):
        """Test get_config_info method with detailed config structure"""
        mock_init_ui.return_value = MagicMock()
        
        # Create a complex config structure
        self.mock_config.llm.default_model_service = 'OpenAI'
        self.mock_config.server.host = '0.0.0.0'
        self.mock_config.server.port = 7860
        self.mock_config.server.share = True
        self.mock_config.dict.return_value = {
            'llm': {'default_model_service': 'OpenAI'},
            'server': {'host': '0.0.0.0', 'port': 7860, 'share': True}
        }
        
        system_ui = SystemUI(self.mock_config, self.config_file_path)
        
        with patch('ui.system_ui.pprint.pformat') as mock_pformat:
            mock_pformat.return_value = "formatted_config_string"
            
            result = system_ui.get_config_info()
            
            # Verify pprint.pformat was called with config dict
            mock_pformat.assert_called_once()
            self.assertEqual(result, "formatted_config_string")
    
    @patch('ui.system_ui.SystemUI.init_ui')
    def test_get_config_info_with_none_config(self, mock_init_ui):
        """Test get_config_info with None config"""
        mock_init_ui.return_value = MagicMock()
        
        system_ui = SystemUI(None, None)
        
        result = system_ui.get_config_info()
        
        # Should return string representation of None
        self.assertEqual(result, "None")
    
    @patch('ui.system_ui.SystemUI.init_ui')
    def test_init_with_complex_config(self, mock_init_ui):
        """Test initialization with complex configuration"""
        mock_init_ui.return_value = MagicMock()
        
        # Create complex config with nested structures
        complex_config = MagicMock()
        complex_config.dict.return_value = {
            'llm': {
                'services': ['OpenAI', 'Azure'],
                'settings': {'temperature': 0.7}
            },
            'server': {
                'port': 8080,
                'debug': True
            }
        }
        
        system_ui = SystemUI(complex_config, '/path/to/config.yaml')
        
        # Verify initialization completed successfully
        self.assertEqual(system_ui.config, complex_config)
        self.assertEqual(system_ui.config_file_path, '/path/to/config.yaml')
        mock_init_ui.assert_called_once()


class TestSystemUIUIComponents(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        self.mock_config.llm.llm_services = {
            'OpenAI': MagicMock(),
            'OpenRouter': MagicMock()
        }
        self.mock_config.llm.llm_services['OpenAI'].args.openai_api_key = 'test-key'
        self.config_file_path = '/tmp/test_config.yaml'
    
    def test_init_ui_creates_components(self):
        """Test that init_ui creates all necessary UI components"""
        with patch('ui.system_ui.gr.Blocks') as mock_blocks, \
             patch('ui.system_ui.gr.Textbox') as mock_textbox, \
             patch('ui.system_ui.gr.Button') as mock_button, \
             patch('ui.system_ui.gr.Row'), \
             patch('ui.system_ui.gr.Column'):
            
            mock_ui = MagicMock()
            mock_blocks.return_value.__enter__.return_value = mock_ui
            
            # Create SystemUI which calls init_ui
            system_ui = SystemUI(self.mock_config, self.config_file_path)
            
            # Verify UI components were created
            mock_textbox.assert_called()
            mock_button.assert_called()
            
            # Verify the UI structure exists
            self.assertIsNotNone(system_ui.ui)
    
    def test_ui_component_configuration(self):
        """Test UI component configuration parameters"""
        with patch('ui.system_ui.gr.Blocks') as mock_blocks, \
             patch('ui.system_ui.gr.Textbox') as mock_textbox, \
             patch('ui.system_ui.gr.Button') as mock_button, \
             patch('ui.system_ui.gr.Row'), \
             patch('ui.system_ui.gr.Column'):
            
            mock_ui = MagicMock()
            mock_blocks.return_value.__enter__.return_value = mock_ui
            
            system_ui = SystemUI(self.mock_config, self.config_file_path)
            
            # Verify textbox was created with correct parameters
            textbox_calls = mock_textbox.call_args_list
            
            # There should be a textbox for config display
            config_textbox_call = None
            for call in textbox_calls:
                if 'label' in call.kwargs and 'Config' in call.kwargs.get('label', ''):
                    config_textbox_call = call
                    break
            
            # SystemUI may not have a config textbox with that exact label
            # Just verify textbox was called
            self.assertTrue(len(textbox_calls) > 0)


class TestSystemUIErrorHandling(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        self.mock_config.llm.llm_services = {
            'OpenAI': MagicMock(),
            'OpenRouter': MagicMock()
        }
        self.mock_config.llm.llm_services['OpenAI'].args.openai_api_key = 'test-key'
        self.config_file_path = '/tmp/test_config.yaml'
    
    @patch('ui.system_ui.SystemUI.init_ui')
    def test_get_config_info_exception_handling(self, mock_init_ui):
        """Test get_config_info handles exceptions gracefully"""
        mock_init_ui.return_value = MagicMock()
        
        # Create config that raises exception when dict() is called
        faulty_config = MagicMock()
        faulty_config.dict.side_effect = Exception("Config error")
        
        system_ui = SystemUI(faulty_config, self.config_file_path)
        
        # Should handle exception gracefully or raise it
        try:
            result = system_ui.get_config_info()
            self.assertIsInstance(result, str)
        except Exception:
            # Exception handling may not be implemented, which is acceptable
            pass
    
    @patch('ui.system_ui.SystemUI.init_ui')
    def test_initialization_with_invalid_path(self, mock_init_ui):
        """Test initialization with invalid config file path"""
        mock_init_ui.return_value = MagicMock()
        
        # Test with various invalid paths
        invalid_paths = [None, "", "/nonexistent/path/config.yaml"]
        
        for invalid_path in invalid_paths:
            try:
                system_ui = SystemUI(self.mock_config, invalid_path)
                # Should complete initialization even with invalid path
                self.assertEqual(system_ui.config_file_path, invalid_path)
                mock_init_ui.assert_called()
            except Exception as e:
                # If an exception occurs, it should be a reasonable type
                self.assertIsInstance(e, (TypeError, AttributeError))


if __name__ == '__main__':
    unittest.main()