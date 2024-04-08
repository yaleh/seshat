import os

from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.llms.replicate import Replicate

class ModelFactory:
    def create_model(self, model_type, model_name, **llm_config):
        if model_type == 'OpenAI':
            return ChatOpenAI(
                model_name=model_name,
                verbose=True,
                **llm_config
            )
        elif model_type == 'Azure_OpenAI':
            return AzureChatOpenAI(
                deployment_name=model_name,
                verbose=True,
                **llm_config
            )
        elif model_type == 'Replicate':
            os.environ['REPLICATE_API_TOKEN'] = llm_config['REPLICATE_API_TOKEN']
            temperature = llm_config['temperature']
            max_tokens=500 if ('max_tokens' not in llm_config or llm_config['max_tokens']=='None') else llm_config['max_tokens']
            return Replicate(
                    model=model_name,
                    model_kwargs={"temperature": temperature if temperature else 0.01, "max_length": max_tokens, "top_p": 1}
                    )
        elif model_type == 'HuggingFace':
            return HuggingFaceHub(
                repo_id=model_name,
                **llm_config
                )
        else:
            raise ValueError("Unsupported chatbot type")
