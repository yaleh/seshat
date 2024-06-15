from typing import Dict, Union

from confz import BaseConfig
from pydantic import BaseModel
from typing import Optional

class OpenAIConfig(BaseModel):
    openai_api_key: str

    class Config:
        extra = "allow"

class AzureOpenAIConfig(BaseModel):
    openai_api_base: str
    openai_api_key: str

    class Config:
        extra = "allow"

class ReplicateConfig(BaseModel):
    REPLICATE_API_TOKEN: str

    class Config:
        extra = "allow"

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
    llm_services: Dict[str, ModelService]

class EmbeddingConfig(BaseModel):
    type: str

    class Config:
        extra = "allow"

class MetaPromptConfig(BaseConfig):
    class Config:
        extra = "allow"

class BatchConfig(BaseConfig):
    class Config:
        extra = "allow"

class ServerConfig(BaseConfig):
    message_db: str
    host: str
    port: int
    share: bool

    class Config:
        extra = "allow"

class AppConfig(BaseConfig):
    llm: LLMConfig
    meta_prompt: MetaPromptConfig
    batch: BatchConfig
    server: ServerConfig
    embedding: Optional[dict[str, EmbeddingConfig]]