import gradio as gr
import pandas as pd
import tempfile

from db.db_sqlite3 import DatabaseManager, LANGSERVE_URLS_TABLE, LANGSERVE_MESSAGES_TABLE
from components.lcel import LLMModelFactory
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage
from langchain.schema.output_parser import StrOutputParser
from tools.table_parser import TableParser
from tools.utils import detect_encoding
from langserve import RemoteRunnable

class LangServeClientUI:
    # def __init__(self, db_manager, llmbot):
    def __init__(self, config,
                 database_name='messages.db',
                 max_message_length=65535
                 ):

        self.config = config
        self.default_model_service = self.config.llm.default_model_service
        self.model_type = self.config.llm.llm_services[self.default_model_service].type
        self.model_name = self.config.llm.llm_services[
            self.default_model_service].default_model
        # self.model_service_list = self.config.llm.llm_services.keys()
        # self.model_name_list = self.config.llm.llm_services[self.default_model_service].models

        self.model_args = self.config.llm.llm_services[self.default_model_service].args

        self.db_manager = DatabaseManager(database_name, max_message_length)
        self.model_factory = LLMModelFactory()

        self.ui = self.init_ui()

    def init_ui(self):
        with gr.Blocks() as batch_ui:
            with gr.Tab('Invoke'):
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            self.message_textbox = gr.Textbox(
                                label="Message",
                                lines=5,
                                show_copy_button=True
                            )
                        self.message_dropdown = gr.Dropdown(
                            choices=self.db_manager.get_messages(LANGSERVE_MESSAGES_TABLE),
                            label="LangServe Message History",
                            allow_custom_value=True,
                            interactive=True
                        )
                        with gr.Group():
                            self.invoke_btn = gr.Button("Invoke", variant="primary")
                    with gr.Column():
                        self.output_textbox = gr.Textbox(
                            label="Output", 
                            lines=5,
                            show_copy_button=True
                        )
                        self.clear_output_btn = gr.Button("Clear Output")

            self.langserve_url = gr.Dropdown(
                choices=self.db_manager.get_messages(LANGSERVE_URLS_TABLE),
                label="LangServer URL", 
                allow_custom_value=True,
                interactive=True
            )

            # Bind events
            self.bind_events()

        return batch_ui

    def bind_events(self):
        self.invoke_btn.click(
            self.invoke, 
            [self.message_textbox, self.langserve_url],
            [self.output_textbox]
        )
        self.output_textbox.change(
            self.update_histories,
            [],
            [self.message_dropdown, self.langserve_url]
        )
        self.message_dropdown.select(
            lambda x: x,
            [self.message_dropdown],
            [self.message_textbox]
        )

    def invoke(self, message, langserver_url):
        if message:
            self.db_manager.append_message(LANGSERVE_MESSAGES_TABLE, message)
        if langserver_url:
            self.db_manager.append_message(LANGSERVE_URLS_TABLE, langserver_url)

        message_dict = eval(message)

        chain = RemoteRunnable(langserver_url)
        output = chain.invoke(message_dict)

        return gr.update(value=output)
    
    def update_histories(self):
        messages = gr.update(
            choices=self.db_manager.get_messages(LANGSERVE_MESSAGES_TABLE),
            interactive=True
            )
        urls = gr.update(
            choices=self.db_manager.get_messages(LANGSERVE_URLS_TABLE),
            interactive=True
            )

        return messages, urls
