import unittest
from unittest.mock import patch, MagicMock, Mock
import gradio as gr

from ui.meta_prompt_ui import MetaPromptUI


class TestMetaPromptUIBasics(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        
        # LLM services configuration
        self.mock_config.llm.default_model_service = 'OpenAI'
        self.mock_config.llm.llm_services = {
            'OpenAI': MagicMock(),
            'Azure_OpenAI': MagicMock()
        }
        
        # OpenAI service configuration
        openai_service = self.mock_config.llm.llm_services['OpenAI']
        openai_service.type = 'OpenAI'
        openai_service.args.dict.return_value = {'openai_api_key': 'test-key'}
        openai_service.models = ['gpt-3.5-turbo', 'gpt-4']
        
        # Azure OpenAI service configuration
        azure_service = self.mock_config.llm.llm_services['Azure_OpenAI']
        azure_service.type = 'Azure_OpenAI'
        azure_service.args.dict.return_value = {'openai_api_key': 'azure-key'}
        azure_service.models = ['gpt-35-turbo', 'gpt-4']
        
        # Meta prompt configuration
        self.mock_config.meta_prompt.default_meta_model_service = 'OpenAI'
        self.mock_config.meta_prompt.default_meta_model_name = 'gpt-4'
        self.mock_config.meta_prompt.default_target_model_service = 'OpenAI'
        self.mock_config.meta_prompt.default_target_model_name = 'gpt-3.5-turbo'
        self.mock_config.meta_prompt.meta_system_prompt = "You are a helpful assistant."
        self.mock_config.meta_prompt.meta_system_prompt_with_other_prompts = "You are a helpful assistant with other prompts."
    
    @patch('ui.meta_prompt_ui.MetaPromptUI.init_ui')
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    def test_meta_prompt_ui_init_basics(self, mock_model_factory, mock_init_ui):
        """Test MetaPromptUI basic initialization"""
        # Setup mocks
        mock_factory_instance = MagicMock()
        mock_model_factory.return_value = mock_factory_instance
        
        mock_ui_instance = MagicMock()
        mock_init_ui.return_value = mock_ui_instance
        
        # Create MetaPromptUI
        meta_ui = MetaPromptUI(self.mock_config)
        
        # Verify initialization
        self.assertEqual(meta_ui.config, self.mock_config)
        self.assertEqual(meta_ui.model_service_list, 'OpenAI')
        
        # Verify generating model configuration
        self.assertEqual(meta_ui.generating_model_service, 'OpenAI')
        self.assertEqual(meta_ui.generating_model_type, 'OpenAI')
        self.assertEqual(meta_ui.generating_model_name, 'gpt-4')
        self.assertEqual(meta_ui.generating_model_args, {'openai_api_key': 'test-key'})
        self.assertFalse(meta_ui.generating_show_other_options)
        
        # Verify testing model configuration
        self.assertEqual(meta_ui.testing_model_service, 'OpenAI')
        self.assertEqual(meta_ui.testing_model_type, 'OpenAI')
        self.assertEqual(meta_ui.testing_model_name, 'gpt-3.5-turbo')
        self.assertEqual(meta_ui.testing_model_args, {'openai_api_key': 'test-key'})
        self.assertFalse(meta_ui.testing_show_other_options)
        
        # Verify other settings
        self.assertFalse(meta_ui.enable_other_user_prompts)
        
        # Verify factories were created
        self.assertEqual(mock_model_factory.call_count, 2)  # chatbot_factory and model_factory
        self.assertEqual(meta_ui.chatbot_factory, mock_factory_instance)
        self.assertEqual(meta_ui.model_factory, mock_factory_instance)
        
        # Verify UI was created
        self.assertEqual(meta_ui.ui, mock_ui_instance)
    
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    def test_meta_prompt_ui_init_with_none_config(self, mock_model_factory):
        """Test MetaPromptUI initialization with None config raises AttributeError"""
        with self.assertRaises(AttributeError):
            MetaPromptUI(None)


class TestMetaPromptUIPromptTesting(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        self.mock_config.llm.default_model_service = 'OpenAI'
        self.mock_config.llm.llm_services = {'OpenAI': MagicMock()}
        openai_service = self.mock_config.llm.llm_services['OpenAI']
        openai_service.type = 'OpenAI'
        openai_service.args.dict.return_value = {'openai_api_key': 'test-key'}
        openai_service.models = ['gpt-3.5-turbo', 'gpt-4']
        
        self.mock_config.meta_prompt.default_meta_model_service = 'OpenAI'
        self.mock_config.meta_prompt.default_meta_model_name = 'gpt-4'
        self.mock_config.meta_prompt.default_target_model_service = 'OpenAI'
        self.mock_config.meta_prompt.default_target_model_name = 'gpt-3.5-turbo'
        self.mock_config.meta_prompt.meta_system_prompt = "You are a helpful assistant."
        self.mock_config.meta_prompt.meta_system_prompt_with_other_prompts = "You are a helpful assistant with other prompts."
    
    @patch('ui.meta_prompt_ui.MetaPromptUI.init_ui')
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    @patch('ui.meta_prompt_ui.SystemMessage')
    @patch('ui.meta_prompt_ui.HumanMessage')
    def test_test_prompt_success(self, mock_human_message, mock_system_message, mock_model_factory, mock_init_ui):
        """Test successful test_prompt operation"""
        # Setup mocks
        mock_init_ui.return_value = MagicMock()
        
        mock_factory_instance = MagicMock()
        mock_model_factory.return_value = mock_factory_instance
        
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Test response"
        mock_llm.invoke.return_value = mock_response
        mock_factory_instance.create_model.return_value = mock_llm
        
        mock_system_msg = MagicMock()
        mock_human_msg = MagicMock()
        mock_system_message.return_value = mock_system_msg
        mock_human_message.return_value = mock_human_msg
        
        # Create MetaPromptUI
        meta_ui = MetaPromptUI(self.mock_config)
        
        # Test test_prompt
        system_prompt = "You are a helpful assistant."
        user_prompt = "What is the capital of France?"
        
        result = meta_ui.test_prompt(system_prompt, user_prompt)
        
        # Verify LLM creation and invocation
        mock_factory_instance.create_model.assert_called_once_with(
            'OpenAI', 'gpt-3.5-turbo', openai_api_key='test-key'
        )
        mock_llm.invoke.assert_called_once()
        
        # Verify messages were created
        mock_system_message.assert_called_once_with(content=system_prompt)
        mock_human_message.assert_called_once_with(content=user_prompt)
        
        self.assertEqual(result, "Test response")
    
    @patch('ui.meta_prompt_ui.MetaPromptUI.init_ui')
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    def test_test_prompt_error(self, mock_model_factory, mock_init_ui):
        """Test test_prompt operation with error"""
        # Setup mocks
        mock_init_ui.return_value = MagicMock()
        
        mock_factory_instance = MagicMock()
        mock_model_factory.return_value = mock_factory_instance
        mock_factory_instance.create_model.side_effect = Exception("LLM creation failed")
        
        # Create MetaPromptUI
        meta_ui = MetaPromptUI(self.mock_config)
        
        # Test test_prompt with exception
        with patch('ui.meta_prompt_ui.gr.Error', side_effect=Exception) as mock_gr_error:
            with self.assertRaises(Exception):
                meta_ui.test_prompt("system prompt", "user prompt")
            
            # Verify gr.Error was called
            mock_gr_error.assert_called_once()


class TestMetaPromptUIMetaPrompt(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        self.mock_config.llm.default_model_service = 'OpenAI'
        self.mock_config.llm.llm_services = {'OpenAI': MagicMock()}
        openai_service = self.mock_config.llm.llm_services['OpenAI']
        openai_service.type = 'OpenAI'
        openai_service.args.dict.return_value = {'openai_api_key': 'test-key'}
        openai_service.models = ['gpt-3.5-turbo', 'gpt-4']
        
        self.mock_config.meta_prompt.default_meta_model_service = 'OpenAI'
        self.mock_config.meta_prompt.default_meta_model_name = 'gpt-4'
        self.mock_config.meta_prompt.default_target_model_service = 'OpenAI'
        self.mock_config.meta_prompt.default_target_model_name = 'gpt-3.5-turbo'
        self.mock_config.meta_prompt.meta_system_prompt = "You are a helpful assistant."
        self.mock_config.meta_prompt.meta_system_prompt_with_other_prompts = "You are a helpful assistant with other prompts."
    
    @patch('ui.meta_prompt_ui.MetaPromptUI.init_ui')
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    @patch('ui.meta_prompt_ui.SystemMessage')
    @patch('ui.meta_prompt_ui.HumanMessage')
    def test_meta_prompt_success_with_extraction(self, mock_human_message, mock_system_message, mock_model_factory, mock_init_ui):
        """Test successful meta_prompt operation with prompt extraction"""
        # Setup mocks
        mock_init_ui.return_value = MagicMock()
        
        mock_factory_instance = MagicMock()
        mock_model_factory.return_value = mock_factory_instance
        
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "<!-- BEGIN OF PROMPT -->Improved system prompt<!-- END OF PROMPT -->"
        mock_llm.invoke.return_value = mock_response
        mock_factory_instance.create_model.return_value = mock_llm
        
        # Create MetaPromptUI
        meta_ui = MetaPromptUI(self.mock_config)
        
        with patch('ui.meta_prompt_ui.gr.Info'):
            # Test meta_prompt
            result_prompt, changed = meta_ui.meta_prompt(
                "meta system prompt",
                "current system prompt", 
                "user prompt",
                "other prompts",
                "expected output",
                "current output",
                True  # use_user_prompts
            )
            
            # Verify results
            self.assertEqual(result_prompt, "Improved system prompt")
            self.assertTrue(changed)  # Should be True since no "NO CHANGE" marker
    
    @patch('ui.meta_prompt_ui.MetaPromptUI.init_ui')
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    def test_meta_prompt_no_change_detected(self, mock_model_factory, mock_init_ui):
        """Test meta_prompt operation when no change is detected"""
        # Setup mocks
        mock_init_ui.return_value = MagicMock()
        
        mock_factory_instance = MagicMock()
        mock_model_factory.return_value = mock_factory_instance
        
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "No changes needed. <!-- NO CHANGE TO PROMPT -->"
        mock_llm.invoke.return_value = mock_response
        mock_factory_instance.create_model.return_value = mock_llm
        
        # Create MetaPromptUI
        meta_ui = MetaPromptUI(self.mock_config)
        
        with patch('ui.meta_prompt_ui.gr.Info'):
            # Test meta_prompt
            result_prompt, changed = meta_ui.meta_prompt(
                "meta system prompt",
                "current system prompt", 
                "user prompt",
                "",
                "expected output",
                "current output",
                False  # use_user_prompts
            )
            
            # Verify results
            self.assertEqual(result_prompt, "No changes needed. <!-- NO CHANGE TO PROMPT -->")
            self.assertFalse(changed)  # Should be False due to "NO CHANGE" marker


class TestMetaPromptUIUtilities(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        self.mock_config.llm.default_model_service = 'OpenAI'
        self.mock_config.llm.llm_services = {'OpenAI': MagicMock()}
        openai_service = self.mock_config.llm.llm_services['OpenAI']
        openai_service.type = 'OpenAI'
        openai_service.args.dict.return_value = {'openai_api_key': 'test-key'}
        openai_service.models = ['gpt-3.5-turbo', 'gpt-4']
        
        self.mock_config.meta_prompt.default_meta_model_service = 'OpenAI'
        self.mock_config.meta_prompt.default_meta_model_name = 'gpt-4'
        self.mock_config.meta_prompt.default_target_model_service = 'OpenAI'
        self.mock_config.meta_prompt.default_target_model_name = 'gpt-3.5-turbo'
        self.mock_config.meta_prompt.meta_system_prompt = "You are a helpful assistant."
        self.mock_config.meta_prompt.meta_system_prompt_with_other_prompts = "You are a helpful assistant with other prompts."
    
    @patch('ui.meta_prompt_ui.MetaPromptUI.init_ui')
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    def test_extract_updated_prompt_success(self, mock_model_factory, mock_init_ui):
        """Test successful prompt extraction"""
        mock_init_ui.return_value = MagicMock()
        
        # Create MetaPromptUI
        meta_ui = MetaPromptUI(self.mock_config)
        
        # Test with valid prompt markers
        gpt_response = "Here is the improved prompt:\n<!-- BEGIN OF PROMPT -->You are an expert assistant.<!-- END OF PROMPT -->\nThat's the updated version."
        result = meta_ui.extract_updated_prompt(gpt_response)
        self.assertEqual(result, "You are an expert assistant.")
        
        # Test with code blocks
        gpt_response_with_code = "<!-- BEGIN OF PROMPT -->```\nYou are an expert assistant.\n```<!-- END OF PROMPT -->"
        result = meta_ui.extract_updated_prompt(gpt_response_with_code)
        self.assertEqual(result, "\nYou are an expert assistant.\n")
    
    @patch('ui.meta_prompt_ui.MetaPromptUI.init_ui')
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    def test_extract_updated_prompt_no_markers(self, mock_model_factory, mock_init_ui):
        """Test prompt extraction when no markers are found"""
        mock_init_ui.return_value = MagicMock()
        
        # Create MetaPromptUI
        meta_ui = MetaPromptUI(self.mock_config)
        
        # Test with no prompt markers
        gpt_response = "This is just regular text without any prompt markers."
        result = meta_ui.extract_updated_prompt(gpt_response)
        self.assertIsNone(result)
    
    @patch('ui.meta_prompt_ui.MetaPromptUI.init_ui')
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    def test_detect_no_change_true(self, mock_model_factory, mock_init_ui):
        """Test no change detection when marker is present"""
        mock_init_ui.return_value = MagicMock()
        
        # Create MetaPromptUI
        meta_ui = MetaPromptUI(self.mock_config)
        
        gpt_response = "The current prompt is already optimal. <!-- NO CHANGE TO PROMPT -->"
        result = meta_ui.detect_no_change(gpt_response)
        self.assertTrue(result)
    
    @patch('ui.meta_prompt_ui.MetaPromptUI.init_ui')
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    def test_detect_no_change_false(self, mock_model_factory, mock_init_ui):
        """Test no change detection when marker is not present"""
        mock_init_ui.return_value = MagicMock()
        
        # Create MetaPromptUI
        meta_ui = MetaPromptUI(self.mock_config)
        
        gpt_response = "Here are some improvements to the prompt."
        result = meta_ui.detect_no_change(gpt_response)
        self.assertFalse(result)
    
    @patch('ui.meta_prompt_ui.MetaPromptUI.init_ui')
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    def test_copy_new_prompts(self, mock_model_factory, mock_init_ui):
        """Test copy_new_prompts method"""
        mock_init_ui.return_value = MagicMock()
        
        # Create MetaPromptUI
        meta_ui = MetaPromptUI(self.mock_config)
        
        system_prompt = "New system prompt"
        output = "New output"
        
        result_prompt, result_output = meta_ui.copy_new_prompts(system_prompt, output)
        
        self.assertEqual(result_prompt, system_prompt)
        self.assertEqual(result_output, output)


class TestMetaPromptUIComparison(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        self.mock_config.llm.default_model_service = 'OpenAI'
        self.mock_config.llm.llm_services = {'OpenAI': MagicMock()}
        openai_service = self.mock_config.llm.llm_services['OpenAI']
        openai_service.type = 'OpenAI'
        openai_service.args.dict.return_value = {'openai_api_key': 'test-key'}
        openai_service.models = ['gpt-3.5-turbo', 'gpt-4']
        
        self.mock_config.meta_prompt.default_meta_model_service = 'OpenAI'
        self.mock_config.meta_prompt.default_meta_model_name = 'gpt-4'
        self.mock_config.meta_prompt.default_target_model_service = 'OpenAI'
        self.mock_config.meta_prompt.default_target_model_name = 'gpt-3.5-turbo'
        self.mock_config.meta_prompt.meta_system_prompt = "You are a helpful assistant."
        self.mock_config.meta_prompt.meta_system_prompt_with_other_prompts = "You are a helpful assistant with other prompts."
    
    @patch('ui.meta_prompt_ui.MetaPromptUI.init_ui')
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    def test_compare_strings_empty_inputs(self, mock_model_factory, mock_init_ui):
        """Test compare_strings with empty inputs"""
        mock_init_ui.return_value = MagicMock()
        
        # Create MetaPromptUI
        meta_ui = MetaPromptUI(self.mock_config)
        
        # Both empty
        result = meta_ui.compare_strings("", "", "expected")
        self.assertIsNone(result)
        
        # Alpha empty
        result = meta_ui.compare_strings("", "beta", "expected")
        self.assertEqual(result, 'B')
        
        # Beta empty
        result = meta_ui.compare_strings("alpha", "", "expected")
        self.assertEqual(result, 'A')
        
        # Both identical
        result = meta_ui.compare_strings("same", "same", "expected")
        self.assertIsNone(result)
    
    @patch('ui.meta_prompt_ui.MetaPromptUI.init_ui')
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    @patch('ui.meta_prompt_ui.CountVectorizer')
    @patch('ui.meta_prompt_ui.cosine_similarity')
    def test_compare_strings_with_similarity(self, mock_cosine_sim, mock_vectorizer, mock_model_factory, mock_init_ui):
        """Test compare_strings with similarity calculation"""
        mock_init_ui.return_value = MagicMock()
        
        # Setup vectorizer mock
        import numpy as np
        mock_vectorizer_instance = MagicMock()
        mock_vectorizer.return_value.fit_transform.return_value = mock_vectorizer_instance
        mock_vectors = np.array([[1, 0], [0, 1], [1, 1]])
        mock_vectorizer_instance.toarray.return_value = mock_vectors
        
        # Setup cosine similarity mock - alpha more similar
        mock_cosine_sim.side_effect = [np.array([[0.8]]), np.array([[0.6]])]
        
        # Create MetaPromptUI
        meta_ui = MetaPromptUI(self.mock_config)
        
        result = meta_ui.compare_strings("alpha", "beta", "expected")
        self.assertEqual(result, 'A')
        
        # Reset mocks and test beta more similar
        mock_cosine_sim.reset_mock()
        mock_cosine_sim.side_effect = [np.array([[0.5]]), np.array([[0.9]])]
        result = meta_ui.compare_strings("alpha", "beta", "expected")
        self.assertEqual(result, 'B')
        
        # Reset mocks and test equal similarity
        mock_cosine_sim.reset_mock()
        mock_cosine_sim.side_effect = [np.array([[0.7]]), np.array([[0.7]])]
        result = meta_ui.compare_strings("alpha", "beta", "expected")
        self.assertIsNone(result)
    
    @patch('ui.meta_prompt_ui.MetaPromptUI.init_ui')
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    def test_delta_similarities_empty_inputs(self, mock_model_factory, mock_init_ui):
        """Test delta_similarities with empty inputs"""
        mock_init_ui.return_value = MagicMock()
        
        # Create MetaPromptUI
        meta_ui = MetaPromptUI(self.mock_config)
        
        # Both empty
        result = meta_ui.delta_similarities("", "", "expected")
        self.assertEqual(result, 0.0)
        
        # Alpha empty
        result = meta_ui.delta_similarities("", "beta", "expected")
        self.assertEqual(result, -1.0)
        
        # Beta empty
        result = meta_ui.delta_similarities("alpha", "", "expected")
        self.assertEqual(result, 1.0)
        
        # Both identical
        result = meta_ui.delta_similarities("same", "same", "expected")
        self.assertEqual(result, 0.0)


class TestMetaPromptUIOptimization(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        self.mock_config.llm.default_model_service = 'OpenAI'
        self.mock_config.llm.llm_services = {'OpenAI': MagicMock()}
        openai_service = self.mock_config.llm.llm_services['OpenAI']
        openai_service.type = 'OpenAI'
        openai_service.args.dict.return_value = {'openai_api_key': 'test-key'}
        openai_service.models = ['gpt-3.5-turbo', 'gpt-4']
        
        self.mock_config.meta_prompt.default_meta_model_service = 'OpenAI'
        self.mock_config.meta_prompt.default_meta_model_name = 'gpt-4'
        self.mock_config.meta_prompt.default_target_model_service = 'OpenAI'
        self.mock_config.meta_prompt.default_target_model_name = 'gpt-3.5-turbo'
        self.mock_config.meta_prompt.meta_system_prompt = "You are a helpful assistant."
        self.mock_config.meta_prompt.meta_prompt_with_other_prompts = "You are a helpful assistant with other prompts."
    
    @patch('ui.meta_prompt_ui.MetaPromptUI.init_ui')
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    def test_optimize_prompt_no_iterations(self, mock_model_factory, mock_init_ui):
        """Test optimize_prompt with zero iterations"""
        mock_init_ui.return_value = MagicMock()
        
        # Create MetaPromptUI
        meta_ui = MetaPromptUI(self.mock_config)
        
        # Create mock progress
        mock_progress = MagicMock()
        mock_progress.tqdm.return_value = []  # No iterations
        
        result_prompt, changed = meta_ui.optimize_prompt(
            "meta prompt", "current prompt", "user prompt", "", 
            "expected", "current output", 0, False, mock_progress
        )
        
        # Should return original prompt with no changes
        self.assertEqual(result_prompt, "current prompt")
        self.assertFalse(changed)
    
    @patch('ui.meta_prompt_ui.MetaPromptUI.init_ui')
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    def test_update_llm_model_name_dropdown(self, mock_model_factory, mock_init_ui):
        """Test update_llm_model_name_dropdown method"""
        mock_init_ui.return_value = MagicMock()
        
        # Create MetaPromptUI
        meta_ui = MetaPromptUI(self.mock_config)
        
        with patch('ui.meta_prompt_ui.gr.Dropdown') as mock_dropdown:
            mock_dropdown_instance = MagicMock()
            mock_dropdown.return_value = mock_dropdown_instance
            
            result = meta_ui.update_llm_model_name_dropdown('OpenAI')
            
            # Verify dropdown was created with correct choices
            mock_dropdown.assert_called_once_with(choices=['gpt-3.5-turbo', 'gpt-4'])
            self.assertEqual(result, mock_dropdown_instance)
    
    @patch('ui.meta_prompt_ui.MetaPromptUI.init_ui')
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    def test_show_other_llm_option(self, mock_model_factory, mock_init_ui):
        """Test show_other_llm_option method"""
        mock_init_ui.return_value = MagicMock()
        
        # Create MetaPromptUI
        meta_ui = MetaPromptUI(self.mock_config)
        
        with patch('ui.meta_prompt_ui.gr.Slider') as mock_slider:
            mock_slider_instances = [MagicMock() for _ in range(4)]
            mock_slider.side_effect = mock_slider_instances
            
            result = meta_ui.show_other_llm_option(True)
            
            # Verify 4 sliders were created
            self.assertEqual(mock_slider.call_count, 4)
            self.assertEqual(len(result), 4)
            
            # Verify all sliders have visible=True
            for call in mock_slider.call_args_list:
                self.assertTrue(call.kwargs['visible'])
    
    @patch('ui.meta_prompt_ui.MetaPromptUI.init_ui')
    @patch('ui.meta_prompt_ui.LLMModelFactory')
    def test_update_enable_other_user_prompts(self, mock_model_factory, mock_init_ui):
        """Test update_enable_other_user_prompts method"""
        mock_init_ui.return_value = MagicMock()
        
        # Create MetaPromptUI
        meta_ui = MetaPromptUI(self.mock_config)
        
        with patch('ui.meta_prompt_ui.gr.Textbox') as mock_textbox:
            mock_textbox_instances = [MagicMock(), MagicMock()]
            mock_textbox.side_effect = mock_textbox_instances
            
            # Test enabling other user prompts
            result = meta_ui.update_enable_other_user_prompts(True)
            
            # Verify attribute was updated
            self.assertTrue(meta_ui.enable_other_user_prompts)
            
            # Verify 2 textboxes were created
            self.assertEqual(mock_textbox.call_count, 2)
            self.assertEqual(len(result), 2)
            
            # Verify first textbox is visible
            first_call = mock_textbox.call_args_list[0]
            self.assertTrue(first_call.kwargs['visible'])


if __name__ == '__main__':
    unittest.main()