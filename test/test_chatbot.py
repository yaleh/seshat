import unittest
import sys
import os
from unittest.mock import patch, MagicMock, Mock
import asyncio
import time

# Add project root to path to import components
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

# Mock the missing prompt module before importing chatbot
sys.modules['components.prompt'] = MagicMock()

from components.chatbot import ChatbotFactory, BaseChatbot, beijing


class TestBeijingTimeFunction(unittest.TestCase):
    
    @patch('components.chatbot.datetime')
    def test_beijing_time_conversion(self, mock_datetime):
        """Test Beijing time conversion function"""
        # Mock the complete datetime chain
        mock_future_time = Mock()
        mock_future_time.timetuple.return_value = "beijing_time_tuple"
        
        mock_now = Mock()
        mock_now.__add__ = Mock(return_value=mock_future_time)
        
        mock_datetime.datetime.now.return_value = mock_now
        mock_datetime.timedelta.return_value = "8_hour_delta"
        
        result = beijing(0, "test")
        
        mock_datetime.datetime.now.assert_called_once()
        mock_datetime.timedelta.assert_called_once_with(hours=8)
        mock_now.__add__.assert_called_once_with("8_hour_delta")
        self.assertEqual(result, "beijing_time_tuple")


class TestChatbotFactory(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.factory = ChatbotFactory()
    
    @patch('components.chatbot.OpenAIChatbot')
    def test_create_openai_chatbot(self, mock_openai_chatbot):
        """Test creating OpenAI chatbot"""
        mock_instance = MagicMock()
        mock_openai_chatbot.return_value = mock_instance
        
        result = self.factory.create_chatbot(
            'OpenAI', 
            'gpt-3.5-turbo', 
            tempreature=0.7,
            max_retries=3,
            request_timeout=60,
            max_tokens=1000,
            api_key='test-key'
        )
        
        mock_openai_chatbot.assert_called_once_with(
            model_name='gpt-3.5-turbo',
            tempreature=0.7,
            max_retries=3,
            request_timeout=60,
            max_tokens=1000,
            api_key='test-key'
        )
        self.assertEqual(result, mock_instance)
    
    @patch('components.chatbot.AzureOpenAIChatbot')
    def test_create_azure_openai_chatbot(self, mock_azure_chatbot):
        """Test creating Azure OpenAI chatbot"""
        mock_instance = MagicMock()
        mock_azure_chatbot.return_value = mock_instance
        
        result = self.factory.create_chatbot(
            'Azure_OpenAI',
            'gpt-35-turbo',
            tempreature=0.5,
            max_retries=5,
            endpoint='https://test.openai.azure.com/'
        )
        
        mock_azure_chatbot.assert_called_once_with(
            model_name='gpt-35-turbo',
            tempreature=0.5,
            max_retries=5,
            request_timeout=600,  # default value
            max_tokens=None,      # default value
            endpoint='https://test.openai.azure.com/'
        )
        self.assertEqual(result, mock_instance)
    
    @patch('components.chatbot.ReplicateChatbot')
    def test_create_replicate_chatbot(self, mock_replicate_chatbot):
        """Test creating Replicate chatbot"""
        mock_instance = MagicMock()
        mock_replicate_chatbot.return_value = mock_instance
        
        result = self.factory.create_chatbot(
            'Replicate',
            'llama-2-7b-chat',
            tempreature=0.8,
            api_token='test-token'
        )
        
        mock_replicate_chatbot.assert_called_once_with(
            model_name='llama-2-7b-chat',
            tempreature=0.8,
            max_retries=6,        # default value
            request_timeout=600,  # default value
            max_tokens=None,      # default value
            api_token='test-token'
        )
        self.assertEqual(result, mock_instance)
    
    def test_create_unsupported_chatbot_type(self):
        """Test creating unsupported chatbot type raises ValueError"""
        with self.assertRaises(ValueError) as context:
            self.factory.create_chatbot('UnsupportedType', 'some-model')
        
        self.assertEqual(str(context.exception), "Unsupported chatbot type")


class TestBaseChatbot(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.chatbot = BaseChatbot(
            temperature=0.7,
            request_timeout=30,
            max_retries=5,
            max_tokens=2000
        )
        # Mock the chat_llm attribute
        self.chatbot.chat_llm = MagicMock()
    
    def test_init_sets_properties(self):
        """Test that initialization sets all properties correctly"""
        self.assertEqual(self.chatbot.temperature, 0.7)
        self.assertEqual(self.chatbot.request_timeout, 30)
        self.assertEqual(self.chatbot.max_retries, 5)
        self.assertEqual(self.chatbot.max_tokens, 2000)
    
    def test_init_with_defaults(self):
        """Test initialization with default values"""
        default_chatbot = BaseChatbot()
        self.assertEqual(default_chatbot.temperature, 0)
        self.assertEqual(default_chatbot.request_timeout, 10)
        self.assertEqual(default_chatbot.max_retries, 3)
        self.assertIsNone(default_chatbot.max_tokens)
    
    def test_call_method(self):
        """Test __call__ method delegates to chat_llm"""
        messages = ["test message"]
        stop = ["stop"]
        callbacks = ["callback"]
        
        self.chatbot(messages, stop=stop, callbacks=callbacks, extra_param="test")
        
        self.chatbot.chat_llm.assert_called_once_with(
            messages=messages,
            stop=stop,
            callbacks=callbacks,
            extra_param="test"
        )
    
    def test_predict_method(self):
        """Test predict method delegates to chat_llm.predict"""
        test_input = "test input"
        expected_output = "test output"
        self.chatbot.chat_llm.predict.return_value = expected_output
        
        result = self.chatbot.predict(test_input)
        
        self.chatbot.chat_llm.predict.assert_called_once_with(test_input)
        self.assertEqual(result, expected_output)
    
    @patch('components.chatbot.PromptCreator')
    @patch('components.chatbot.time')
    @patch('components.chatbot.logging')
    def test_qa_answer_question(self, mock_logging, mock_time, mock_prompt_creator):
        """Test qa_answer_question method"""
        # Setup mocks
        mock_time.time.side_effect = [100.0, 102.5]  # start and end times
        mock_prompt = "formatted prompt"
        mock_prompt_creator.create_prompt.return_value = mock_prompt
        
        mock_response = MagicMock()
        mock_response.content = "AI response"
        self.chatbot.chat_llm.return_value = mock_response
        
        # Test data
        system_prompt = "You are a helpful assistant"
        user_prompt = "What is AI?"
        history = [("previous question", "previous answer")]
        
        result = self.chatbot.qa_answer_question(system_prompt, user_prompt, history)
        
        # Verify prompt creation
        mock_prompt_creator.create_prompt.assert_called_once_with(
            system_prompt, user_prompt, history
        )
        
        # Verify chat_llm call
        self.chatbot.chat_llm.assert_called_once_with(mock_prompt)
        
        # Verify history update
        expected_history = [
            ("previous question", "previous answer"),
            (user_prompt, "AI response")
        ]
        self.assertEqual(result, expected_history)
        
        # Verify logging
        mock_logging.info.assert_called_once()
        log_call_args = mock_logging.info.call_args[0][0]
        self.assertIn("Request ID:", log_call_args)
        self.assertIn("GPT Process time: 2.50", log_call_args)
    
    @patch('components.chatbot.PromptCreator')
    @patch('components.chatbot.asyncio')
    @patch('components.chatbot.time')
    @patch('components.chatbot.logging')
    def test_batch_send_async_loop(self, mock_logging, mock_time, mock_asyncio, mock_prompt_creator):
        """Test batch_send_async_loop method"""
        # Setup mocks
        mock_time.time.side_effect = [200.0, 205.0]  # start and end times
        
        mock_prompts = [MagicMock(), MagicMock()]
        mock_prompts[0].messages = "prompt1_messages"
        mock_prompts[1].messages = "prompt2_messages"
        mock_prompt_creator.create_table_prompts.return_value = mock_prompts
        
        mock_loop = MagicMock()
        mock_asyncio.new_event_loop.return_value = mock_loop
        mock_loop.run_until_complete.return_value = ["response1", "response2"]
        
        self.chatbot.chat_llm.abatch = MagicMock()
        
        # Test data
        system_prompt = "Process this data"
        user_prompt = "Analyze: {data}"
        history = []
        table = [{"data": "row1"}, {"data": "row2"}]
        
        result = self.chatbot.batch_send_async_loop(system_prompt, user_prompt, history, table)
        
        # Verify prompt creation
        mock_prompt_creator.create_table_prompts.assert_called_once_with(
            system_prompt, user_prompt, table
        )
        
        # Verify async loop management
        mock_asyncio.new_event_loop.assert_called_once()
        mock_asyncio.set_event_loop.assert_called_once_with(mock_loop)
        mock_loop.run_until_complete.assert_called_once()
        mock_loop.close.assert_called_once()
        
        # Verify result structure (returns gpt_result, prompts)
        self.assertEqual(len(result), 2)  # responses, prompts
        self.assertEqual(result[0], ["response1", "response2"])  # responses
        self.assertEqual(result[1], mock_prompts)               # prompts
    
    @patch('components.chatbot.PromptCreator')
    @patch('components.chatbot.asyncio')
    def test_batch_send_async_loop_exception_handling(self, mock_asyncio, mock_prompt_creator):
        """Test that batch_send_async_loop properly handles exceptions"""
        # Setup mocks
        mock_prompts = [MagicMock()]
        mock_prompt_creator.create_table_prompts.return_value = mock_prompts
        
        mock_loop = MagicMock()
        mock_asyncio.new_event_loop.return_value = mock_loop
        mock_loop.run_until_complete.side_effect = Exception("Test exception")
        
        self.chatbot.chat_llm.abatch = MagicMock()
        
        # Test that exception is raised but loop is still closed
        with self.assertRaises(Exception):
            self.chatbot.batch_send_async_loop("sys", "user", [], [{"data": "test"}])
        
        # Verify loop was still closed despite exception
        mock_loop.close.assert_called_once()


    @patch('components.chatbot.PromptCreator')
    @patch('components.chatbot.time')
    @patch('components.chatbot.logging')
    def test_batch_send(self, mock_logging, mock_time, mock_prompt_creator):
        """Test batch_send method"""
        # Setup mocks
        mock_time.time.side_effect = [300.0, 303.0]  # start and end times
        
        mock_prompts = [MagicMock(), MagicMock()]
        mock_prompts[0].messages = "prompt1_messages"
        mock_prompts[1].messages = "prompt2_messages"
        mock_prompt_creator.create_table_prompts.return_value = mock_prompts
        
        mock_batch_result = ["batch_response1", "batch_response2"]
        self.chatbot.chat_llm.batch = MagicMock(return_value=mock_batch_result)
        
        # Test data
        system_prompt = "Process this data"
        user_prompt = "Analyze: {data}"
        table = [{"data": "row1"}, {"data": "row2"}]
        
        result = self.chatbot.batch_send(system_prompt, user_prompt, table)
        
        # Verify prompt creation
        mock_prompt_creator.create_table_prompts.assert_called_once_with(
            system_prompt, user_prompt, table
        )
        
        # Verify batch call
        self.chatbot.chat_llm.batch.assert_called_once_with(
            ["prompt1_messages", "prompt2_messages"]
        )
        
        # Verify result
        self.assertEqual(result[0], mock_batch_result)
        self.assertEqual(result[1], mock_prompts)
        
        # Verify logging
        mock_logging.info.assert_called_once()


class TestOpenAIChatbot(unittest.TestCase):
    
    @patch('components.chatbot.ChatOpenAI')
    @patch('components.chatbot.ConversationChain')
    @patch('components.chatbot.ConversationBufferMemory')
    def test_openai_chatbot_init(self, mock_memory, mock_conversation, mock_chat_openai):
        """Test OpenAIChatbot initialization"""
        mock_chat_instance = MagicMock()
        mock_chat_openai.return_value = mock_chat_instance
        mock_memory_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        mock_conversation_instance = MagicMock()
        mock_conversation.return_value = mock_conversation_instance
        
        llm_config = {
            'openai_api_base': 'https://api.openai.com/v1',
            'openai_api_key': 'test-key'
        }
        
        from components.chatbot import OpenAIChatbot
        chatbot = OpenAIChatbot(
            model_name='gpt-4',
            tempreature=0.8,
            max_retries=5,
            request_timeout=120,
            max_tokens=2000,
            **llm_config
        )
        
        # Verify ChatOpenAI was called correctly
        mock_chat_openai.assert_called_once_with(
            model_name='gpt-4',
            openai_api_base='https://api.openai.com/v1',
            openai_api_key='test-key',
            temperature=0.8,
            max_retries=5,
            request_timeout=120,
            max_tokens=2000,
            verbose=True
        )
        
        # Verify memory and conversation setup
        mock_memory.assert_called_once()
        mock_conversation.assert_called_once_with(
            llm=mock_chat_instance,
            verbose=True,
            memory=mock_memory_instance
        )
    
    @patch('components.chatbot.ChatOpenAI')
    @patch('components.chatbot.ConversationChain')
    @patch('components.chatbot.ConversationBufferMemory')
    def test_openai_chatbot_init_none_max_tokens(self, mock_memory, mock_conversation, mock_chat_openai):
        """Test OpenAIChatbot initialization with max_tokens='None'"""
        llm_config = {
            'openai_api_base': 'https://api.openai.com/v1',
            'openai_api_key': 'test-key'
        }
        
        from components.chatbot import OpenAIChatbot
        chatbot = OpenAIChatbot(max_tokens='None', **llm_config)
        
        # Verify max_tokens was converted to None
        call_args = mock_chat_openai.call_args[1]
        self.assertIsNone(call_args['max_tokens'])
    
    @patch('components.chatbot.ChatOpenAI')
    @patch('components.chatbot.ConversationChain')
    @patch('components.chatbot.ConversationBufferMemory')
    def test_openai_chatbot_conversation_method(self, mock_memory, mock_conversation, mock_chat_openai):
        """Test OpenAIChatbot conversaction method"""
        # Setup mocks
        mock_memory_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        mock_memory_instance.load_memory_variables.return_value = {'history': 'line1\nline2\nline3'}
        
        mock_conversation_instance = MagicMock()
        mock_conversation.return_value = mock_conversation_instance
        mock_conversation_instance.return_value = "AI response"
        
        llm_config = {
            'openai_api_base': 'https://api.openai.com/v1',
            'openai_api_key': 'test-key'
        }
        
        from components.chatbot import OpenAIChatbot
        chatbot = OpenAIChatbot(**llm_config)
        
        # Test the conversation method
        result = chatbot.conversaction("test input")
        
        # Verify memory loading and conversation call
        mock_memory_instance.load_memory_variables.assert_called_once_with({})
        mock_conversation_instance.assert_called_once_with("test input")


class TestAzureOpenAIChatbot(unittest.TestCase):
    
    @patch('components.chatbot.AzureChatOpenAI')
    def test_azure_openai_chatbot_init(self, mock_azure_chat):
        """Test AzureOpenAIChatbot initialization"""
        mock_chat_instance = MagicMock()
        mock_azure_chat.return_value = mock_chat_instance
        
        llm_config = {
            'openai_api_type': 'azure',
            'openai_api_version': '2023-05-15',
            'openai_api_base': 'https://test.openai.azure.com/',
            'openai_api_key': 'test-key'
        }
        
        from components.chatbot import AzureOpenAIChatbot
        chatbot = AzureOpenAIChatbot(
            model_name='gpt-35-turbo',
            tempreature=0.7,
            max_retries=3,
            request_timeout=60,
            max_tokens=1500,
            **llm_config
        )
        
        # Verify AzureChatOpenAI was called correctly
        mock_azure_chat.assert_called_once_with(
            deployment_name='gpt-35-turbo',
            openai_api_type='azure',
            openai_api_version='2023-05-15',
            openai_api_base='https://test.openai.azure.com/',
            openai_api_key='test-key',
            temperature=0.7,
            max_retries=3,
            request_timeout=60,
            max_tokens=1500,
            verbose=True
        )


class TestReplicateChatbot(unittest.TestCase):
    
    @patch('components.chatbot.Replicate')
    def test_replicate_chatbot_init(self, mock_replicate):
        """Test ReplicateChatbot initialization"""
        mock_replicate_instance = MagicMock()
        mock_replicate.return_value = mock_replicate_instance
        
        llm_config = {
            'REPLICATE_API_TOKEN': 'test-token'
        }
        
        from components.chatbot import ReplicateChatbot
        chatbot = ReplicateChatbot(
            model_name='custom-model',
            tempreature=0.9,
            max_tokens=1000,
            **llm_config
        )
        
        # Verify environment variable was set
        self.assertEqual(os.environ.get('REPLICATE_API_TOKEN'), 'test-token')
        
        # Verify Replicate was called correctly
        mock_replicate.assert_called_once_with(
            model='custom-model',
            model_kwargs={
                "temperature": 0.9,
                "max_length": 1000,
                "top_p": 1
            }
        )
    
    @patch('components.chatbot.Replicate')
    def test_replicate_chatbot_call_method(self, mock_replicate):
        """Test ReplicateChatbot __call__ method"""
        mock_replicate_instance = MagicMock()
        mock_replicate_instance.return_value = "replicate response"
        mock_replicate.return_value = mock_replicate_instance
        
        llm_config = {'REPLICATE_API_TOKEN': 'test-token'}
        
        from components.chatbot import ReplicateChatbot
        chatbot = ReplicateChatbot(**llm_config)
        
        # Test the __call__ method
        result = chatbot(
            messages="test message",
            stop=["stop"],
            callbacks=["callback"],
            extra_param="test"
        )
        
        # Verify the call was forwarded correctly
        mock_replicate_instance.assert_called_once_with(
            prompt="test message",
            stop=["stop"],
            extra_param="test"
        )


class TestChatbotIntegration(unittest.TestCase):
    
    @patch('components.chatbot.PromptCreator')
    def test_chatbot_workflow_integration(self, mock_prompt_creator):
        """Test integration of chatbot workflow components"""
        # Create a chatbot with mocked chat_llm
        chatbot = BaseChatbot()
        chatbot.chat_llm = MagicMock()
        
        # Mock prompt creator
        mock_prompt_creator.create_prompt.return_value = "test prompt"
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "Test response"
        chatbot.chat_llm.return_value = mock_response
        
        # Test the workflow
        history = []
        result = chatbot.qa_answer_question(
            "You are helpful",
            "What is 2+2?",
            history
        )
        
        # Verify the complete workflow
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ("What is 2+2?", "Test response"))


if __name__ == '__main__':
    unittest.main()