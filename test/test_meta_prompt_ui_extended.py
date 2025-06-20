import unittest
import pandas as pd
import tempfile
from unittest.mock import patch, MagicMock, Mock
import gradio as gr

from ui.meta_prompt_ui import MetaPromptUI


class TestMetaPromptUIBasics(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        self.mock_config.llm.default_model_service = 'OpenAI'
        self.mock_config.llm.llm_services = {
            'OpenAI': MagicMock()
        }
        self.mock_config.llm.llm_services['OpenAI'].type = 'OpenAI'
        self.mock_config.llm.llm_services['OpenAI'].args = MagicMock()
        self.mock_config.llm.llm_services['OpenAI'].args.dict.return_value = {'openai_api_key': 'test-key'}
        
        # Mock meta_prompt config
        self.mock_config.meta_prompt.default_meta_model_service = 'OpenAI'
        self.mock_config.meta_prompt.default_meta_model_name = 'gpt-4'
        self.mock_config.meta_prompt.default_target_model_service = 'OpenAI'
        self.mock_config.meta_prompt.default_target_model_name = 'gpt-3.5-turbo'
    
    @patch('ui.meta_prompt_ui.MetaPromptUI.init_ui')
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    def test_meta_prompt_ui_init_basics(self, mock_llm_factory, mock_init_ui):
        """Test MetaPromptUI basic initialization"""
        # Setup mocks
        mock_factory_instance = MagicMock()
        mock_llm_factory.return_value = mock_factory_instance
        
        mock_ui_instance = MagicMock()
        mock_init_ui.return_value = mock_ui_instance
        
        # Create MetaPromptUI
        meta_ui = MetaPromptUI(self.mock_config)
        
        # Verify initialization
        self.assertEqual(meta_ui.config, self.mock_config)
        self.assertEqual(meta_ui.model_service_list, 'OpenAI')
        self.assertEqual(meta_ui.generating_model_service, 'OpenAI')
        self.assertEqual(meta_ui.generating_model_type, 'OpenAI')
        self.assertEqual(meta_ui.generating_model_name, 'gpt-4')
        self.assertEqual(meta_ui.testing_model_service, 'OpenAI')
        self.assertEqual(meta_ui.testing_model_type, 'OpenAI')
        self.assertEqual(meta_ui.testing_model_name, 'gpt-3.5-turbo')
        
        # Verify factories were created
        self.assertEqual(mock_llm_factory.call_count, 2)  # chatbot_factory and model_factory
        self.assertEqual(meta_ui.chatbot_factory, mock_factory_instance)
        self.assertEqual(meta_ui.model_factory, mock_factory_instance)
        
        # Verify UI was created
        self.assertEqual(meta_ui.ui, mock_ui_instance)
    
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    def test_meta_prompt_ui_init_with_none_config(self, mock_llm_factory):
        """Test MetaPromptUI initialization with None config raises AttributeError"""
        with self.assertRaises(AttributeError):
            MetaPromptUI(None)


class TestMetaPromptUIConfiguration(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        self.mock_config.llm.default_model_service = 'Azure_OpenAI'
        self.mock_config.llm.llm_services = {
            'Azure_OpenAI': MagicMock()
        }
        self.mock_config.llm.llm_services['Azure_OpenAI'].type = 'Azure_OpenAI'
        self.mock_config.llm.llm_services['Azure_OpenAI'].args = MagicMock()
        self.mock_config.llm.llm_services['Azure_OpenAI'].args.dict.return_value = {
            'openai_api_key': 'azure-key',
            'openai_api_base': 'https://test.openai.azure.com/'
        }
        
        # Mock meta_prompt config with different services
        self.mock_config.meta_prompt.default_meta_model_service = 'Azure_OpenAI'
        self.mock_config.meta_prompt.default_meta_model_name = 'gpt-4'
        self.mock_config.meta_prompt.default_target_model_service = 'Azure_OpenAI'
        self.mock_config.meta_prompt.default_target_model_name = 'gpt-35-turbo'
    
    @patch('ui.meta_prompt_ui.MetaPromptUI.init_ui')
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    def test_meta_prompt_ui_with_azure_config(self, mock_llm_factory, mock_init_ui):
        """Test MetaPromptUI with Azure OpenAI configuration"""
        # Setup mocks
        mock_factory_instance = MagicMock()
        mock_llm_factory.return_value = mock_factory_instance
        mock_init_ui.return_value = MagicMock()
        
        # Create MetaPromptUI
        meta_ui = MetaPromptUI(self.mock_config)
        
        # Verify Azure configuration is properly loaded
        self.assertEqual(meta_ui.generating_model_service, 'Azure_OpenAI')
        self.assertEqual(meta_ui.generating_model_type, 'Azure_OpenAI')
        self.assertEqual(meta_ui.generating_model_name, 'gpt-4')
        self.assertEqual(meta_ui.testing_model_service, 'Azure_OpenAI')
        self.assertEqual(meta_ui.testing_model_type, 'Azure_OpenAI')
        self.assertEqual(meta_ui.testing_model_name, 'gpt-35-turbo')
        
        # Verify model args contain Azure-specific settings
        expected_args = {
            'openai_api_key': 'azure-key',
            'openai_api_base': 'https://test.openai.azure.com/'
        }
        self.assertEqual(meta_ui.generating_model_args, expected_args)
        self.assertEqual(meta_ui.testing_model_args, expected_args)


class TestMetaPromptUIPromptGeneration(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        self.mock_config.llm.default_model_service = 'OpenAI'
        self.mock_config.llm.llm_services = {'OpenAI': MagicMock()}
        self.mock_config.llm.llm_services['OpenAI'].type = 'OpenAI'
        self.mock_config.llm.llm_services['OpenAI'].args = MagicMock()
        self.mock_config.llm.llm_services['OpenAI'].args.dict.return_value = {'openai_api_key': 'test-key'}
        
        self.mock_config.meta_prompt.default_meta_model_service = 'OpenAI'
        self.mock_config.meta_prompt.default_meta_model_name = 'gpt-4'
        self.mock_config.meta_prompt.default_target_model_service = 'OpenAI'
        self.mock_config.meta_prompt.default_target_model_name = 'gpt-3.5-turbo'
    
    @patch('ui.meta_prompt_ui.MetaPromptUI.init_ui')
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    @patch('ui.meta_prompt_ui.SystemMessage')
    @patch('ui.meta_prompt_ui.HumanMessage')
    def test_generate_prompt_success(self, mock_human_message, mock_system_message, mock_llm_factory, mock_init_ui):
        """Test successful prompt generation"""
        # Setup mocks
        mock_factory_instance = MagicMock()
        mock_llm_factory.return_value = mock_factory_instance
        mock_init_ui.return_value = MagicMock()
        
        # Mock LLM chain
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "Generated system prompt"
        mock_factory_instance.create_model.return_value = mock_llm
        
        # Mock message objects
        mock_system_msg = MagicMock()
        mock_human_msg = MagicMock()
        mock_system_message.return_value = mock_system_msg
        mock_human_message.return_value = mock_human_msg
        
        # Create MetaPromptUI
        meta_ui = MetaPromptUI(self.mock_config)
        
        # Test generate_prompt
        user_prompt = "Test user prompt"
        expected_output = "Expected output"
        other_prompts = "```prompt1```\n```prompt2```"
        
        with patch('ui.meta_prompt_ui.gr.update') as mock_gr_update:
            mock_gr_update.side_effect = ['prompt_update', 'changed_update']
            
            prompt_result, changed_result = meta_ui.generate_prompt(
                user_prompt, expected_output, other_prompts, 1
            )
            
            # Verify model creation
            mock_factory_instance.create_model.assert_called_with(
                model_type='OpenAI',
                model_name='gpt-4',
                openai_api_key='test-key'
            )
            
            # Verify messages were created
            mock_system_message.assert_called()
            mock_human_message.assert_called()
            
            # Verify LLM invocation
            mock_llm.invoke.assert_called()
            
            # Verify gr.update calls
            self.assertEqual(mock_gr_update.call_count, 2)
            self.assertEqual(prompt_result, 'prompt_update')
            self.assertEqual(changed_result, 'changed_update')
    
    @patch('ui.meta_prompt_ui.MetaPromptUI.init_ui')
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    def test_generate_prompt_error(self, mock_llm_factory, mock_init_ui):
        """Test prompt generation with error"""
        # Setup mocks
        mock_factory_instance = MagicMock()
        mock_factory_instance.create_model.side_effect = Exception("Model creation failed")
        mock_llm_factory.return_value = mock_factory_instance
        mock_init_ui.return_value = MagicMock()
        
        # Create MetaPromptUI
        meta_ui = MetaPromptUI(self.mock_config)
        
        # Test generate_prompt with exception
        with patch('ui.meta_prompt_ui.gr.Error', side_effect=Exception) as mock_gr_error:
            with self.assertRaises(Exception):
                meta_ui.generate_prompt("test", "test", "", 1)
            
            # Verify gr.Error was called
            mock_gr_error.assert_called_once()


class TestMetaPromptUIPromptTesting(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        self.mock_config.llm.default_model_service = 'OpenAI'
        self.mock_config.llm.llm_services = {'OpenAI': MagicMock()}
        self.mock_config.llm.llm_services['OpenAI'].type = 'OpenAI'
        self.mock_config.llm.llm_services['OpenAI'].args = MagicMock()
        self.mock_config.llm.llm_services['OpenAI'].args.dict.return_value = {'openai_api_key': 'test-key'}
        
        self.mock_config.meta_prompt.default_meta_model_service = 'OpenAI'
        self.mock_config.meta_prompt.default_meta_model_name = 'gpt-4'
        self.mock_config.meta_prompt.default_target_model_service = 'OpenAI'
        self.mock_config.meta_prompt.default_target_model_name = 'gpt-3.5-turbo'
    
    @patch('ui.meta_prompt_ui.MetaPromptUI.init_ui')
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    @patch('ui.meta_prompt_ui.SystemMessage')
    @patch('ui.meta_prompt_ui.HumanMessage')
    def test_test_prompt_success(self, mock_human_message, mock_system_message, mock_llm_factory, mock_init_ui):
        """Test successful prompt testing"""
        # Setup mocks
        mock_factory_instance = MagicMock()
        mock_llm_factory.return_value = mock_factory_instance
        mock_init_ui.return_value = MagicMock()
        
        # Mock LLM chain
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "Test output"
        mock_factory_instance.create_model.return_value = mock_llm
        
        # Mock message objects
        mock_system_msg = MagicMock()
        mock_human_msg = MagicMock()
        mock_system_message.return_value = mock_system_msg
        mock_human_message.return_value = mock_human_msg
        
        # Create MetaPromptUI
        meta_ui = MetaPromptUI(self.mock_config)
        
        # Test test_prompt
        system_prompt = "Test system prompt"
        user_prompt = "Test user prompt"
        
        with patch('ui.meta_prompt_ui.gr.update') as mock_gr_update:
            mock_gr_update.return_value = 'output_update'
            
            result = meta_ui.test_prompt(system_prompt, user_prompt)
            
            # Verify model creation with testing model
            mock_factory_instance.create_model.assert_called_with(
                model_type='OpenAI',
                model_name='gpt-3.5-turbo',
                openai_api_key='test-key'
            )
            
            # Verify messages were created
            mock_system_message.assert_called_once_with(content=system_prompt)
            mock_human_message.assert_called_once_with(content=user_prompt)
            
            # Verify LLM invocation
            mock_llm.invoke.assert_called_once_with([mock_system_msg, mock_human_msg])
            
            # Verify gr.update call
            mock_gr_update.assert_called_once_with(value="Test output")
            self.assertEqual(result, 'output_update')
    
    @patch('ui.meta_prompt_ui.MetaPromptUI.init_ui')
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    def test_test_prompt_error(self, mock_llm_factory, mock_init_ui):
        """Test prompt testing with error"""
        # Setup mocks
        mock_factory_instance = MagicMock()
        mock_factory_instance.create_model.side_effect = Exception("Model error")
        mock_llm_factory.return_value = mock_factory_instance
        mock_init_ui.return_value = MagicMock()
        
        # Create MetaPromptUI
        meta_ui = MetaPromptUI(self.mock_config)
        
        # Test test_prompt with exception
        with patch('ui.meta_prompt_ui.gr.Error', side_effect=Exception) as mock_gr_error:
            with self.assertRaises(Exception):
                meta_ui.test_prompt("system", "user")
            
            # Verify gr.Error was called
            mock_gr_error.assert_called_once()


class TestMetaPromptUIOptimization(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        self.mock_config.llm.default_model_service = 'OpenAI'
        self.mock_config.llm.llm_services = {'OpenAI': MagicMock()}
        self.mock_config.llm.llm_services['OpenAI'].type = 'OpenAI'
        self.mock_config.llm.llm_services['OpenAI'].args = MagicMock()
        self.mock_config.llm.llm_services['OpenAI'].args.dict.return_value = {'openai_api_key': 'test-key'}
        
        self.mock_config.meta_prompt.default_meta_model_service = 'OpenAI'
        self.mock_config.meta_prompt.default_meta_model_name = 'gpt-4'
        self.mock_config.meta_prompt.default_target_model_service = 'OpenAI'
        self.mock_config.meta_prompt.default_target_model_name = 'gpt-3.5-turbo'
    
    @patch('ui.meta_prompt_ui.MetaPromptUI.init_ui')
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    @patch('ui.meta_prompt_ui.MetaPromptUI.generate_prompt')
    @patch('ui.meta_prompt_ui.MetaPromptUI.test_prompt')
    def test_optimize_prompt_multiple_iterations(self, mock_test_prompt, mock_generate_prompt, mock_llm_factory, mock_init_ui):
        """Test prompt optimization with multiple iterations"""
        # Setup mocks
        mock_factory_instance = MagicMock()
        mock_llm_factory.return_value = mock_factory_instance
        mock_init_ui.return_value = MagicMock()
        
        # Mock generate_prompt and test_prompt methods
        mock_generate_prompt.return_value = ('new_prompt_update', 'changed_update')
        mock_test_prompt.return_value = 'test_output_update'
        
        # Create MetaPromptUI
        meta_ui = MetaPromptUI(self.mock_config)
        
        # Test optimize_prompt with 3 iterations
        user_prompt = "Test user prompt"
        expected_output = "Expected output"
        other_prompts = ""
        iterations = 3
        
        with patch('ui.meta_prompt_ui.gr.Info') as mock_gr_info:
            prompt_result, changed_result, output_result = meta_ui.optimize_prompt(
                user_prompt, expected_output, other_prompts, iterations
            )
            
            # Verify generate_prompt was called for each iteration
            self.assertEqual(mock_generate_prompt.call_count, iterations)
            
            # Verify test_prompt was called for each iteration
            self.assertEqual(mock_test_prompt.call_count, iterations)
            
            # Verify info message was shown
            mock_gr_info.assert_called_with(f"Completed {iterations} optimization iterations")
            
            # Verify final results
            self.assertEqual(prompt_result, 'new_prompt_update')
            self.assertEqual(changed_result, 'changed_update')
            self.assertEqual(output_result, 'test_output_update')
    
    @patch('ui.meta_prompt_ui.MetaPromptUI.init_ui')
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    @patch('ui.meta_prompt_ui.MetaPromptUI.generate_prompt')
    def test_optimize_prompt_error_handling(self, mock_generate_prompt, mock_llm_factory, mock_init_ui):
        """Test prompt optimization error handling"""
        # Setup mocks
        mock_factory_instance = MagicMock()
        mock_llm_factory.return_value = mock_factory_instance
        mock_init_ui.return_value = MagicMock()
        
        # Mock generate_prompt to raise exception
        mock_generate_prompt.side_effect = Exception("Generation failed")
        
        # Create MetaPromptUI
        meta_ui = MetaPromptUI(self.mock_config)
        
        # Test optimize_prompt with exception
        with patch('ui.meta_prompt_ui.gr.Error', side_effect=Exception) as mock_gr_error:
            with self.assertRaises(Exception):
                meta_ui.optimize_prompt("test", "test", "", 1)
            
            # Verify gr.Error was called
            mock_gr_error.assert_called_once()


class TestMetaPromptUIUtilities(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        self.mock_config.llm.default_model_service = 'OpenAI'
        self.mock_config.llm.llm_services = {'OpenAI': MagicMock()}
        self.mock_config.llm.llm_services['OpenAI'].type = 'OpenAI'
        self.mock_config.llm.llm_services['OpenAI'].args = MagicMock()
        self.mock_config.llm.llm_services['OpenAI'].args.dict.return_value = {'openai_api_key': 'test-key'}
        
        self.mock_config.meta_prompt.default_meta_model_service = 'OpenAI'
        self.mock_config.meta_prompt.default_meta_model_name = 'gpt-4'
        self.mock_config.meta_prompt.default_target_model_service = 'OpenAI'
        self.mock_config.meta_prompt.default_target_model_name = 'gpt-3.5-turbo'
    
    @patch('ui.meta_prompt_ui.MetaPromptUI.init_ui')
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    def test_toggle_other_user_prompts(self, mock_llm_factory, mock_init_ui):
        """Test toggle_other_user_prompts method"""
        # Setup mocks
        mock_factory_instance = MagicMock()
        mock_llm_factory.return_value = mock_factory_instance
        mock_init_ui.return_value = MagicMock()
        
        # Create MetaPromptUI
        meta_ui = MetaPromptUI(self.mock_config)
        
        # Test toggling other user prompts
        with patch('ui.meta_prompt_ui.gr.update') as mock_gr_update:
            mock_gr_update.return_value = 'visibility_update'
            
            # Test enabling
            result = meta_ui.toggle_other_user_prompts(True)
            self.assertEqual(meta_ui.enable_other_user_prompts, True)
            mock_gr_update.assert_called_with(visible=True)
            self.assertEqual(result, 'visibility_update')
            
            # Test disabling
            result = meta_ui.toggle_other_user_prompts(False)
            self.assertEqual(meta_ui.enable_other_user_prompts, False)
            mock_gr_update.assert_called_with(visible=False)
            self.assertEqual(result, 'visibility_update')
    
    @patch('ui.meta_prompt_ui.MetaPromptUI.init_ui')
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    def test_utility_functions_existence(self, mock_llm_factory, mock_init_ui):
        """Test that utility functions exist and can be called"""
        # Setup mocks
        mock_factory_instance = MagicMock()
        mock_llm_factory.return_value = mock_factory_instance
        mock_init_ui.return_value = MagicMock()
        
        # Create MetaPromptUI
        meta_ui = MetaPromptUI(self.mock_config)
        
        # Verify utility methods exist
        self.assertTrue(hasattr(meta_ui, 'toggle_other_user_prompts'))
        self.assertTrue(hasattr(meta_ui, 'generate_prompt'))
        self.assertTrue(hasattr(meta_ui, 'test_prompt'))
        self.assertTrue(hasattr(meta_ui, 'optimize_prompt'))
        
        # Verify initial state
        self.assertFalse(meta_ui.enable_other_user_prompts)
        self.assertFalse(meta_ui.generating_show_other_options)
        self.assertFalse(meta_ui.testing_show_other_options)


class TestMetaPromptUIUIComponents(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        self.mock_config.llm.default_model_service = 'OpenAI'
        self.mock_config.llm.llm_services = {'OpenAI': MagicMock()}
        self.mock_config.llm.llm_services['OpenAI'].type = 'OpenAI'
        self.mock_config.llm.llm_services['OpenAI'].args = MagicMock()
        self.mock_config.llm.llm_services['OpenAI'].args.dict.return_value = {'openai_api_key': 'test-key'}
        
        self.mock_config.meta_prompt.default_meta_model_service = 'OpenAI'
        self.mock_config.meta_prompt.default_meta_model_name = 'gpt-4'
        self.mock_config.meta_prompt.default_target_model_service = 'OpenAI'
        self.mock_config.meta_prompt.default_target_model_name = 'gpt-3.5-turbo'
    
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    def test_init_ui_creates_components(self, mock_llm_factory):
        """Test that init_ui creates all necessary UI components"""
        mock_llm_factory.return_value = MagicMock()
        
        with patch('ui.meta_prompt_ui.gr.Blocks') as mock_blocks, \
             patch('ui.meta_prompt_ui.gr.Row'), \
             patch('ui.meta_prompt_ui.gr.Column'), \
             patch('ui.meta_prompt_ui.gr.Textbox') as mock_textbox, \
             patch('ui.meta_prompt_ui.gr.Checkbox') as mock_checkbox, \
             patch('ui.meta_prompt_ui.gr.Number') as mock_number, \
             patch('ui.meta_prompt_ui.gr.Button') as mock_button:
            
            mock_ui = MagicMock()
            mock_blocks.return_value.__enter__.return_value = mock_ui
            
            # Create MetaPromptUI which calls init_ui
            meta_ui = MetaPromptUI(self.mock_config)
            
            # Verify UI components were created
            mock_textbox.assert_called()
            mock_checkbox.assert_called()
            mock_number.assert_called()
            mock_button.assert_called()
            
            # Verify the UI structure exists
            self.assertIsNotNone(meta_ui.ui)
    
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    def test_ui_component_configuration(self, mock_llm_factory):
        """Test UI component configuration parameters"""
        mock_llm_factory.return_value = MagicMock()
        
        with patch('ui.meta_prompt_ui.gr.Blocks') as mock_blocks, \
             patch('ui.meta_prompt_ui.gr.Row'), \
             patch('ui.meta_prompt_ui.gr.Column'), \
             patch('ui.meta_prompt_ui.gr.Textbox') as mock_textbox, \
             patch('ui.meta_prompt_ui.gr.Checkbox') as mock_checkbox, \
             patch('ui.meta_prompt_ui.gr.Number') as mock_number, \
             patch('ui.meta_prompt_ui.gr.Button') as mock_button:
            
            mock_ui = MagicMock()
            mock_blocks.return_value.__enter__.return_value = mock_ui
            
            meta_ui = MetaPromptUI(self.mock_config)
            
            # Verify textbox components were created with appropriate parameters
            textbox_calls = mock_textbox.call_args_list
            self.assertTrue(len(textbox_calls) > 0)
            
            # Check for specific textboxes
            labels_found = []
            for call in textbox_calls:
                if 'label' in call.kwargs:
                    labels_found.append(call.kwargs['label'])
            
            # Should have textboxes for user prompt, expected output, new system prompt, etc.
            expected_labels = ['Testing User Prompt', 'Expected Output', 'New System Prompt', 'New Output']
            for label in expected_labels:
                self.assertIn(label, labels_found)


if __name__ == '__main__':
    unittest.main()