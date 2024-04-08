from typing import Dict, Union
import yaml
from confz import BaseConfig
from pydantic import BaseModel, SecretStr, Extra

class ConfigLoader():
    def __init__(self, config_path) -> None:
        self.llm_config = self.get_default_config(config_path)

    def get_default_config(self, config_path='config/llm_config.yaml'):
        with open(config_path) as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def get_model_config(self, model_service):
        return self.llm_config["MODEL"]['MODEL_SERVICE'][model_service]['CONFIG']

    def get_default_model_service(self):
        return self.llm_config["MODEL"]['DEFAULT_MODEL_SERVICE']

    def get_model_type(self, model_service):
        return self.llm_config["MODEL"]['MODEL_SERVICE'][model_service]['TYPE']

    def get_model_service_list(self):
        return list(self.llm_config["MODEL"]['MODEL_SERVICE'].keys())

    def get_default_model_name(self, default_model_service):
        return self.llm_config["MODEL"]['MODEL_SERVICE'][default_model_service]['DEFAULT_MODEL_NAME']

    def get_model_name_list(self, model_service):
        return list(self.llm_config["MODEL"]['MODEL_SERVICE'][model_service]['MODEL_NAME'])
    
    def get_default_rr_model_name(self):
        return self.llm_config["MODEL"]['DEFAULT_RR_MODEL_NAME']

class OpenAIConfig(BaseModel):
    # openai_api_base: str
    openai_api_key: str
    # temperature: int
    # max_retries: int
    # request_timeout: int
    # max_tokens: int = None

    class Config:
        extra = Extra.allow

class AzureOpenAIConfig(BaseModel):
    openai_api_type: str
    openai_api_version: str
    openai_api_base: str
    openai_api_key: str
    temperature: int
    max_retries: int
    request_timeout: int
    max_tokens: int = None

class ReplicateConfig(BaseModel):
    REPLICATE_API_TOKEN: str
    temperature: int
    max_tokens: int = None

class HuggingFaceConfig(BaseModel):
    huggingfacehub_api_token: str
    # temperature: float
    # max_length: int
    
    class Config:
        extra = Extra.allow

class ModelService(BaseModel):
    type: str
    default_model: str
    models: list[str]
    args: Union[OpenAIConfig, AzureOpenAIConfig, ReplicateConfig, HuggingFaceConfig]

class LLMConfig(BaseModel):
    default_model_service: str
    default_rr_model_name: str
    model_services: Dict[str, ModelService]

class AppConfig(BaseConfig):
    llm: LLMConfig