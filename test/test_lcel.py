import unittest
import os
from unittest.mock import patch, MagicMock

from components.lcel import LLMModelFactory, EmbeddingFactory


class TestLLMModelFactory(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.factory = LLMModelFactory()
    
    @patch('components.lcel.ChatOpenAI')
    def test_create_openai_model(self, mock_chat_openai):
        """Test creating OpenAI model"""
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance
        
        config = {
            'openai_api_key': 'test-key',
            'temperature': 0.7,
            'max_tokens': 1000
        }
        
        result = self.factory.create_model('OpenAI', 'gpt-3.5-turbo', **config)
        
        mock_chat_openai.assert_called_once_with(
            model_name='gpt-3.5-turbo',
            verbose=True,
            **config
        )
        self.assertEqual(result, mock_instance)
    
    @patch('components.lcel.AzureChatOpenAI')
    def test_create_azure_openai_model(self, mock_azure_chat_openai):
        """Test creating Azure OpenAI model"""
        mock_instance = MagicMock()
        mock_azure_chat_openai.return_value = mock_instance
        
        config = {
            'openai_api_key': 'test-key',
            'openai_api_base': 'https://test.openai.azure.com/',
            'openai_api_version': '2023-05-15'
        }
        
        result = self.factory.create_model('Azure_OpenAI', 'gpt-35-turbo', **config)
        
        mock_azure_chat_openai.assert_called_once_with(
            deployment_name='gpt-35-turbo',
            verbose=True,
            **config
        )
        self.assertEqual(result, mock_instance)
    
    @patch('components.lcel.Replicate')
    def test_create_replicate_model(self, mock_replicate):
        """Test creating Replicate model"""
        mock_instance = MagicMock()
        mock_replicate.return_value = mock_instance
        
        config = {
            'REPLICATE_API_TOKEN': 'test-token',
            'temperature': 0.8,
            'max_tokens': 2000
        }
        
        result = self.factory.create_model('Replicate', 'test-model', **config)
        
        # Check environment variable was set
        self.assertEqual(os.environ.get('REPLICATE_API_TOKEN'), 'test-token')
        
        # Check Replicate was called with correct parameters
        mock_replicate.assert_called_once_with(
            model='test-model',
            model_kwargs={
                "temperature": 0.8,
                "max_length": 2000,
                "top_p": 1
            }
        )
        self.assertEqual(result, mock_instance)
    
    @patch('components.lcel.Replicate')
    def test_create_replicate_model_default_max_tokens(self, mock_replicate):
        """Test creating Replicate model with default max_tokens"""
        mock_instance = MagicMock()
        mock_replicate.return_value = mock_instance
        
        config = {
            'REPLICATE_API_TOKEN': 'test-token',
            'temperature': 0.5
        }
        
        result = self.factory.create_model('Replicate', 'test-model', **config)
        
        mock_replicate.assert_called_once_with(
            model='test-model',
            model_kwargs={
                "temperature": 0.5,
                "max_length": 500,  # Default value
                "top_p": 1
            }
        )
    
    @patch('components.lcel.Replicate')
    def test_create_replicate_model_max_tokens_none(self, mock_replicate):
        """Test creating Replicate model with max_tokens='None'"""
        mock_instance = MagicMock()
        mock_replicate.return_value = mock_instance
        
        config = {
            'REPLICATE_API_TOKEN': 'test-token',
            'temperature': 0.5,
            'max_tokens': 'None'
        }
        
        result = self.factory.create_model('Replicate', 'test-model', **config)
        
        mock_replicate.assert_called_once_with(
            model='test-model',
            model_kwargs={
                "temperature": 0.5,
                "max_length": 500,  # Default when 'None'
                "top_p": 1
            }
        )
    
    @patch('components.lcel.Replicate')
    def test_create_replicate_model_no_temperature(self, mock_replicate):
        """Test creating Replicate model without temperature"""
        mock_instance = MagicMock()
        mock_replicate.return_value = mock_instance
        
        config = {
            'REPLICATE_API_TOKEN': 'test-token'
        }
        
        # This should raise a KeyError since temperature is required
        with self.assertRaises(KeyError):
            result = self.factory.create_model('Replicate', 'test-model', **config)
    
    @patch('components.lcel.HuggingFaceHub')
    def test_create_huggingface_model(self, mock_huggingface_hub):
        """Test creating HuggingFace model"""
        mock_instance = MagicMock()
        mock_huggingface_hub.return_value = mock_instance
        
        config = {
            'huggingfacehub_api_token': 'test-token',
            'model_kwargs': {'temperature': 0.5, 'max_length': 64}
        }
        
        result = self.factory.create_model('HuggingFace', 'gpt2', **config)
        
        mock_huggingface_hub.assert_called_once_with(
            repo_id='gpt2',
            **config
        )
        self.assertEqual(result, mock_instance)
    
    def test_create_unsupported_model_type(self):
        """Test creating unsupported model type raises ValueError"""
        with self.assertRaises(ValueError) as context:
            self.factory.create_model('UnsupportedType', 'some-model')
        
        self.assertEqual(str(context.exception), "Unsupported chatbot type")


class TestEmbeddingFactory(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.factory = EmbeddingFactory()
    
    @patch('components.lcel.OpenAIEmbeddings')
    def test_create_openai_embeddings(self, mock_openai_embeddings):
        """Test creating OpenAI embeddings"""
        mock_instance = MagicMock()
        mock_openai_embeddings.return_value = mock_instance
        
        # Mock globals() to include OpenAIEmbeddings
        with patch('components.lcel.globals') as mock_globals:
            mock_globals.return_value = {'OpenAIEmbeddings': mock_openai_embeddings}
            
            kwargs = {
                'openai_api_key': 'test-key',
                'model': 'text-embedding-3-small'
            }
            
            result = self.factory.create('OpenAIEmbeddings', **kwargs)
            
            mock_openai_embeddings.assert_called_once_with(**kwargs)
            self.assertEqual(result, mock_instance)
    
    @patch('components.lcel.AzureOpenAIEmbeddings')
    def test_create_azure_openai_embeddings(self, mock_azure_embeddings):
        """Test creating Azure OpenAI embeddings"""
        mock_instance = MagicMock()
        mock_azure_embeddings.return_value = mock_instance
        
        # Mock globals() to include AzureOpenAIEmbeddings
        with patch('components.lcel.globals') as mock_globals:
            mock_globals.return_value = {'AzureOpenAIEmbeddings': mock_azure_embeddings}
            
            kwargs = {
                'openai_api_key': 'test-key',
                'azure_endpoint': 'https://test.openai.azure.com/',
                'model': 'text-embedding-3-small'
            }
            
            result = self.factory.create('AzureOpenAIEmbeddings', **kwargs)
            
            mock_azure_embeddings.assert_called_once_with(**kwargs)
            self.assertEqual(result, mock_instance)
    
    @patch('components.lcel.HuggingFaceHubEmbeddings')
    def test_create_huggingface_embeddings(self, mock_hf_embeddings):
        """Test creating HuggingFace embeddings"""
        mock_instance = MagicMock()
        mock_hf_embeddings.return_value = mock_instance
        
        # Mock globals() to include HuggingFaceHubEmbeddings
        with patch('components.lcel.globals') as mock_globals:
            mock_globals.return_value = {'HuggingFaceHubEmbeddings': mock_hf_embeddings}
            
            kwargs = {
                'model': 'sentence-transformers/all-MiniLM-L6-v2'
            }
            
            result = self.factory.create('HuggingFaceHubEmbeddings', **kwargs)
            
            mock_hf_embeddings.assert_called_once_with(**kwargs)
            self.assertEqual(result, mock_instance)
    
    def test_create_nonexistent_embedding_type(self):
        """Test creating non-existent embedding type raises KeyError"""
        with patch('components.lcel.globals') as mock_globals:
            mock_globals.return_value = {}  # Empty globals
            
            with self.assertRaises(KeyError):
                self.factory.create('NonExistentEmbedding', model='test')


if __name__ == '__main__':
    unittest.main()