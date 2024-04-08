from typing import Dict, Union

from confz import BaseConfig
from pydantic import BaseModel

class OpenAIConfig(BaseModel):
    openai_api_key: str

    class Config:
        extra = "allow"

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

    class Config:
        extra = "allow"

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