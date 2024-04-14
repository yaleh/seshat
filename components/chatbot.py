import time
import datetime
import asyncio
import logging
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import Replicate
import sys
import os
# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将项目根目录添加到Python路径
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from components.prompt import PromptCreator
import datetime

REQUEST_ID = 0
def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()

logging.Formatter.converter = beijing
logging.basicConfig(filename='mimir.log',
                    format="%(asctime)s:[%(levelname)s]:%(name)s:%(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.DEBUG)

class ChatbotFactory:
    def create_chatbot(self, chatbot_type, model_name, tempreature=0, max_retries=6, request_timeout=600, max_tokens=None, **llm_config):
        if chatbot_type == 'OpenAI':
            return OpenAIChatbot(model_name=model_name,
                                 tempreature=tempreature,
                                 max_retries=max_retries,
                                 request_timeout=request_timeout,
                                 max_tokens=max_tokens,
                                 **llm_config)
        elif chatbot_type == 'Azure_OpenAI':
            return AzureOpenAIChatbot(model_name=model_name,
                                      tempreature=tempreature,
                                      max_retries=max_retries,
                                      request_timeout=request_timeout,
                                      max_tokens=max_tokens,
                                      **llm_config)
        elif chatbot_type == 'Replicate':
            return ReplicateChatbot(model_name=model_name,
                                      tempreature=tempreature,
                                      max_retries=max_retries,
                                      request_timeout=request_timeout,
                                      max_tokens=max_tokens,
                                      **llm_config)
        else:
            raise ValueError("Unsupported chatbot type")


class BaseChatbot:
    def __init__(self, temperature=0, request_timeout=10, max_retries=3, max_tokens=None):
        self.temperature=temperature
        self.request_timeout=request_timeout
        self.max_retries=max_retries
        self.max_tokens = max_tokens

    def __call__(self, messages, stop=None, callbacks=None, **kwargs):
        return self.chat_llm(messages=messages, stop=stop, callbacks=callbacks, **kwargs)

    def predict(self, input):
        return self.chat_llm.predict(input)

    def qa_answer_question(self, system_prompt, user_prompt, history):
        prompt = PromptCreator.create_prompt(system_prompt, user_prompt, history)
        timestamp_start = time.time()
        gpt_response = self.chat_llm(prompt)
        timestamp_end = time.time()
        timestamp_diff = timestamp_end - timestamp_start
        global REQUEST_ID
        REQUEST_ID += 1
        logging.info(f"Request ID: {REQUEST_ID} \n \
                    Runnable : __call__ \n \
                    Prompt: {prompt} \n \
                    GPT Response: {gpt_response} \n \
                    GPT Process time: {timestamp_diff:.2f}")
        history.append((user_prompt, gpt_response.content))
        return history

    def batch_send_async_loop(self, system_prompt, user_prompt, history, table):
        prompts = PromptCreator.create_table_prompts(system_prompt, user_prompt, table)
        # run prompts in async mode with self.chat_llm.abatch
        # Create a new event loop
        loop = asyncio.new_event_loop()
        # Set the loop as the default for the current context
        asyncio.set_event_loop(loop)
        timestamp_start = time.time()
        try:
            gpt_result = loop.run_until_complete(self.chat_llm.abatch([prompt.messages for prompt in prompts]))
        finally:
            loop.close()
        timestamp_end = time.time()
        timestamp_diff = timestamp_end - timestamp_start
        global REQUEST_ID
        REQUEST_ID += 1
        logging.info(f"Request ID: {REQUEST_ID} \n \
                    Runnable : abatch(async/loop) \n \
                    Prompts: {prompts} \n \
                    GPT Response: {gpt_result} \n \
                    GPT Process time: {timestamp_diff:.2f}")
        # for response, prompt in zip(result, prompts):
        #     history.append((prompt.messages[-1].content, response.content))
        return gpt_result, prompts

    def batch_send(self, system_prompt, user_prompt, table):
        prompts = PromptCreator.create_table_prompts(system_prompt, user_prompt, table)
        timestamp_start = time.time()
        gpt_result = self.chat_llm.batch([prompt.messages for prompt in prompts])
        timestamp_end = time.time()
        timestamp_diff = timestamp_end - timestamp_start
        global REQUEST_ID
        REQUEST_ID += 1
        logging.info(f"Request ID: {REQUEST_ID} \n \
                    Runnable : batch \n \
                    Prompts: {prompts} \n \
                    GPT Response: {gpt_result} \n \
                    GPT Process time: {timestamp_diff:.2f}")
        # for response, prompt in zip(result, prompts):
        #     history.append((prompt.messages[-1].content, response.content))
        # return history
        return gpt_result, prompts

class OpenAIChatbot(BaseChatbot):
    def __init__(self, model_name='gpt-3.5-turbo', tempreature=0, max_retries=6, request_timeout=600, max_tokens=None, **llm_config):
        super().__init__()
        if max_tokens=='None':
            max_tokens = None
            
        self.chat_llm = ChatOpenAI(
            model_name=model_name,
            openai_api_base=llm_config['openai_api_base'],
            openai_api_key=llm_config['openai_api_key'],
            temperature=tempreature,
            max_retries=max_retries,
            request_timeout=request_timeout,
            max_tokens=max_tokens if max_tokens else None,
            verbose=True
        )
        self.memory = ConversationBufferMemory()
        self.conversation = ConversationChain(
            llm=self.chat_llm, verbose=True, memory=self.memory)

    def conversaction(self, input):
        history = self.memory.load_memory_variables({})['history'].split('\n')
        res = self.conversation(input)
        return history.append(res)


class AzureOpenAIChatbot(BaseChatbot):
    def __init__(self, model_name='ChatGpt', tempreature=0, max_retries=6, request_timeout=600, max_tokens=None, **llm_config):
        super().__init__()
        self.chat_llm = AzureChatOpenAI(
            deployment_name=model_name,
            openai_api_type=llm_config['openai_api_type'],
            openai_api_version=llm_config['openai_api_version'],
            openai_api_base=llm_config['openai_api_base'],
            openai_api_key=llm_config['openai_api_key'],
            temperature=tempreature,
            max_retries=max_retries,
            request_timeout=request_timeout,
            max_tokens=max_tokens if max_tokens else None,
            verbose=True
        )

class ReplicateChatbot(BaseChatbot):
    def __init__(self, model_name='a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', tempreature=0, max_retries=6, request_timeout=600, max_tokens=None, **llm_config):
        super().__init__()
        os.environ['REPLICATE_API_TOKEN'] = llm_config['REPLICATE_API_TOKEN']
        self.chat_llm = Replicate(
                model=model_name,
                model_kwargs={"temperature": tempreature if tempreature else 0.01, "max_length": max_tokens if max_tokens else 500, "top_p": 1}
                )

    def __call__(self, messages, stop=None, callbacks=None, **kwargs):
        return self.chat_llm(prompt=messages, stop=stop, **kwargs)

if __name__ == "__main__":
    import sys
    import os
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 将项目根目录添加到Python路径
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    sys.path.append(project_root)
    from tools.config_loader import ConfigLoader
    config_loader = ConfigLoader(config_path='configs/custom_llm_config.yaml')
    chatbot_factory = ChatbotFactory()
    # OpenAI
    # llm_config = config_loader.get_model_config('OpenAI')
    # OpenAI_chatbot = chatbot_factory.create_chatbot(chatbot_type='OpenAI', model_name='gpt-3.5-turbo', tempreature=0, max_retries=6, request_timeout=600, **llm_config)
    # print(OpenAI_chatbot.predict('hello'))
    # Azure_OpenAI
    # azure_llm_config = config_loader.get_model_config('Azure_OpenAI')
    # Azure_OpenAI_chatbot = chatbot_factory.create_chatbot(chatbot_type='Azure_OpenAI', model_name='ChatGpt', **azure_llm_config)
    # print(Azure_OpenAI_chatbot.predict('hello'))
    # Replicate
    replicate_llm_config = config_loader.get_model_config('Replicate')
    Replicate_chatbot = chatbot_factory.create_chatbot(chatbot_type='Replicate', model_name='a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', **replicate_llm_config)
    print(Replicate_chatbot.predict('tell me a joke about lions'))
# end main