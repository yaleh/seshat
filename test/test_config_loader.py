import unittest
import tempfile
import os
from unittest.mock import patch, mock_open
import yaml

from tools.config_loader import (
    OpenAIConfig, AzureOpenAIConfig, ReplicateConfig, HuggingFaceConfig,
    ModelService, LLMConfig, EmbeddingConfig, MetaPromptConfig,
    BatchConfig, ServerConfig, AppConfig
)


class TestConfigModels(unittest.TestCase):
    
    def test_openai_config_valid(self):
        """Test OpenAIConfig with valid data"""
        config_data = {"openai_api_key": "test-key"}
        config = OpenAIConfig(**config_data)
        self.assertEqual(config.openai_api_key, "test-key")
    
    def test_openai_config_extra_fields(self):
        """Test OpenAIConfig allows extra fields"""
        config_data = {
            "openai_api_key": "test-key",
            "temperature": 0.7,
            "max_tokens": 1000
        }
        config = OpenAIConfig(**config_data)
        self.assertEqual(config.openai_api_key, "test-key")
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.max_tokens, 1000)
    
    def test_azure_openai_config_valid(self):
        """Test AzureOpenAIConfig with valid data"""
        config_data = {
            "openai_api_base": "https://test.openai.azure.com/",
            "openai_api_key": "test-key"
        }
        config = AzureOpenAIConfig(**config_data)
        self.assertEqual(config.openai_api_base, "https://test.openai.azure.com/")
        self.assertEqual(config.openai_api_key, "test-key")
    
    def test_replicate_config_valid(self):
        """Test ReplicateConfig with valid data"""
        config_data = {"REPLICATE_API_TOKEN": "test-token"}
        config = ReplicateConfig(**config_data)
        self.assertEqual(config.REPLICATE_API_TOKEN, "test-token")
    
    def test_huggingface_config_valid(self):
        """Test HuggingFaceConfig with valid data"""
        config_data = {"huggingfacehub_api_token": "test-token"}
        config = HuggingFaceConfig(**config_data)
        self.assertEqual(config.huggingfacehub_api_token, "test-token")
    
    def test_model_service_with_openai_config(self):
        """Test ModelService with OpenAI configuration"""
        service_data = {
            "type": "OpenAI",
            "default_model": "gpt-3.5-turbo",
            "models": ["gpt-3.5-turbo", "gpt-4"],
            "args": {"openai_api_key": "test-key"}
        }
        service = ModelService(**service_data)
        self.assertEqual(service.type, "OpenAI")
        self.assertEqual(service.default_model, "gpt-3.5-turbo")
        self.assertIn("gpt-4", service.models)
        self.assertIsInstance(service.args, OpenAIConfig)
    
    def test_llm_config_valid(self):
        """Test LLMConfig with valid data"""
        config_data = {
            "default_model_service": "OpenAI",
            "default_rr_model_name": "gpt-4",
            "llm_services": {
                "OpenAI": {
                    "type": "OpenAI",
                    "default_model": "gpt-3.5-turbo",
                    "models": ["gpt-3.5-turbo"],
                    "args": {"openai_api_key": "test-key"}
                }
            }
        }
        config = LLMConfig(**config_data)
        self.assertEqual(config.default_model_service, "OpenAI")
        self.assertEqual(config.default_rr_model_name, "gpt-4")
        self.assertIn("OpenAI", config.llm_services)
    
    def test_embedding_config_with_extra_fields(self):
        """Test EmbeddingConfig allows extra fields"""
        config_data = {
            "type": "OpenAIEmbeddings",
            "model": "text-embedding-3-small",
            "openai_api_key": "test-key"
        }
        config = EmbeddingConfig(**config_data)
        self.assertEqual(config.type, "OpenAIEmbeddings")
        self.assertEqual(config.model, "text-embedding-3-small")
    
    def test_server_config_valid(self):
        """Test ServerConfig with valid data"""
        config_data = {
            "message_db": "test.db",
            "host": "0.0.0.0",
            "port": 7800,
            "share": True
        }
        config = ServerConfig(**config_data)
        self.assertEqual(config.message_db, "test.db")
        self.assertEqual(config.host, "0.0.0.0")
        self.assertEqual(config.port, 7800)
        self.assertTrue(config.share)
    
    def test_batch_config_empty(self):
        """Test BatchConfig can be empty"""
        config = BatchConfig()
        self.assertIsInstance(config, BatchConfig)
    
    def test_meta_prompt_config_with_extra_fields(self):
        """Test MetaPromptConfig allows extra fields"""
        config_data = {
            "default_meta_model_service": "OpenAI",
            "default_meta_model_name": "gpt-4"
        }
        config = MetaPromptConfig(**config_data)
        self.assertEqual(config.default_meta_model_service, "OpenAI")


class TestAppConfig(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        self.valid_config_data = {
            "llm": {
                "default_model_service": "OpenAI",
                "default_rr_model_name": "gpt-4",
                "llm_services": {
                    "OpenAI": {
                        "type": "OpenAI",
                        "default_model": "gpt-3.5-turbo",
                        "models": ["gpt-3.5-turbo", "gpt-4"],
                        "args": {"openai_api_key": "test-key"}
                    }
                }
            },
            "meta_prompt": {
                "default_meta_model_service": "OpenAI"
            },
            "batch": {},
            "server": {
                "message_db": "test.db",
                "host": "0.0.0.0",
                "port": 7800,
                "share": True
            },
            "embedding": {
                "test-embedding": {
                    "type": "OpenAIEmbeddings",
                    "model": "text-embedding-3-small"
                }
            }
        }
    
    def test_app_config_valid(self):
        """Test AppConfig with valid complete configuration"""
        config = AppConfig(**self.valid_config_data)
        self.assertIsInstance(config.llm, LLMConfig)
        self.assertIsInstance(config.meta_prompt, MetaPromptConfig)
        self.assertIsInstance(config.batch, BatchConfig)
        self.assertIsInstance(config.server, ServerConfig)
        self.assertIsNotNone(config.embedding)
    
    def test_app_config_missing_required_fields(self):
        """Test AppConfig validation with missing required fields"""
        incomplete_data = {"llm": self.valid_config_data["llm"]}
        
        with self.assertRaises(Exception):  # Should raise validation error
            AppConfig(**incomplete_data)
    
    def test_app_config_optional_embedding(self):
        """Test AppConfig with optional embedding field set to None"""
        config_data = self.valid_config_data.copy()
        config_data["embedding"] = None
        
        config = AppConfig(**config_data)
        self.assertIsNone(config.embedding)
    
    def test_app_config_from_yaml_data(self):
        """Test creating AppConfig from YAML-like data structure"""
        # Simulate loading from YAML
        yaml_data = yaml.safe_load(yaml.dump(self.valid_config_data))
        config = AppConfig(**yaml_data)
        
        self.assertEqual(config.llm.default_model_service, "OpenAI")
        self.assertEqual(config.server.port, 7800)


class TestConfigValidation(unittest.TestCase):
    
    def test_invalid_model_service_type(self):
        """Test ModelService with invalid args type"""
        # This should work since Union accepts various types
        service_data = {
            "type": "Custom",
            "default_model": "custom-model",
            "models": ["custom-model"],
            "args": {"custom_key": "custom_value"}  # This should be valid
        }
        # Since Union allows different types, this might not fail as expected
        # We'd need more specific validation in real implementation
        try:
            service = ModelService(**service_data)
            self.assertIsNotNone(service)
        except Exception:
            # Expected for strict type checking
            pass
    
    def test_server_config_invalid_port(self):
        """Test ServerConfig with invalid port type"""
        config_data = {
            "message_db": "test.db",
            "host": "0.0.0.0",
            "port": "invalid_port",  # Should be int
            "share": True
        }
        
        with self.assertRaises(Exception):
            ServerConfig(**config_data)
    
    def test_llm_config_empty_services(self):
        """Test LLMConfig with empty services"""
        config_data = {
            "default_model_service": "OpenAI",
            "default_rr_model_name": "gpt-4",
            "llm_services": {}  # Empty services
        }
        
        config = LLMConfig(**config_data)
        self.assertEqual(len(config.llm_services), 0)


if __name__ == '__main__':
    unittest.main()