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
import pprint

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
                        self.clear_invoke_output_btn = gr.Button("Clear Output")
            with gr.Tab("Batch"):
                with gr.Row():
                    with gr.Column():
                        self.input_dataframe = gr.Dataframe(
                            label="Table Dataframe for Input",
                            # value=[],
                            headers=['input'],
                            wrap=True,
                            interactive=True
                        )
                        # with gr.Group():
                        #     self.update_batch_message_btn = gr.Button("Update Batch Message")
                        #     self.batch_message = gr.Textbox(
                        #         label="Batch Message",
                        #         lines=5,
                        #         show_copy_button=True
                        #     )
                        # with gr.Row():
                        #     self.table_rows = gr.Number(label="Table Rows", value=0, precision=0, 
                        #                                 interactive=False)
                            # self.refresh_table_rows = gr.Button(value="Refresh Table Rows")
                        with gr.Row():
                            with gr.Group():
                                self.input_file = gr.File(
                                    label="Input file(*.csv/xls/xlsx)",
                                    file_types=["csv", "xls", "xlsx"],
                                    height=120
                                )
                                self.update_input_file_btn = gr.Button(value="Update")
                            with gr.Column():
                                self.table_rows = gr.Number(label="Table Rows", value=0, precision=0,
                                                            interactive=False)
                                self.batch_start = gr.Number(
                                    label="Batch Start", value=0, minimum=0, precision=0)
                                self.batch_end = gr.Number(
                                    label="Batch End (excluded)", value=0, minimum=0, precision=0)
                                self.batch_size = gr.Number(
                                    label="Batch Size", value=10, minimum=0, precision=0)
                        with gr.Row():
                            self.batch_button = gr.Button(value="Batch", variant='primary')
                            self.cancel_batch_button = gr.Button(value="Cancel Batch")
                    with gr.Column():
                        with gr.Row():
                            self.output_dataframe = gr.Dataframe(
                                label="Table Dataframe for Output",
                                value=[],
                                wrap=True,
                                interactive=True
                            )
                        with gr.Row():
                            self.output_file = gr.File(
                                        label='Download Result Files',
                                        file_count='multiple',
                                        file_types=['csv', 'xls', 'xlsx'],
                                        interactive=False
                                    )
                        with gr.Row():
                            self.clear_batch_output = gr.ClearButton(
                                [self.output_dataframe, self.output_file],
                                value="Clear Output"
                            )
                            self.clear_all_batch = gr.ClearButton(
                                [
                                    self.input_dataframe,
                                    # self.batch_message,
                                    self.input_file,
                                    self.output_dataframe,
                                    self.output_file
                                ],
                                value="Clear All"
                            )
        

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
        # self.update_batch_message_btn.click(
        #     self.update_batch_message,
        #     [self.input_dataframe],
        #     [self.batch_message]
        # )
        self.input_file.upload(
            self.upload_table,
            [self.input_file],
            [self.input_dataframe]
        )
        self.update_input_file_btn.click(
            self.update_input_files,
            [self.input_dataframe],
            [self.input_file]
        )
        self.batch_button.click(
            self.batch,
            [self.input_dataframe, self.batch_start, self.batch_end, self.batch_size, self.langserve_url],
            [self.output_dataframe]
        )
        self.output_dataframe.change(
            self.update_output_files,
            [self.output_dataframe],
            [self.output_file]
        )
        self.input_dataframe.change(
            self.update_input_rows,
            [self.input_dataframe],
            [self.table_rows, self.batch_start, self.batch_end, self.batch_size]
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

    def update_batch_message(self, table_dataframe_input):
        if table_dataframe_input is not None:
            _ = _dataframe_to_batch_message(table_dataframe_input)
            return gr.update(value=_)
        return gr.update(value='')
    
    def upload_table(self, file):
        try:
            if file.name.endswith('csv'):
                enco = detect_encoding(file.name)
                df = pd.read_csv(file.name, encoding=enco)
            elif file.name.endswith('xls') or file.name.endswith('xlsx'):
                df = pd.read_excel(file.name)
            else:
                df = pd.DataFrame()
        except Exception as e:
            raise gr.Error(e)
        # df = TableParser.add_skip_column(df)
        df_size = len(df)
        return df

    def batch(self, table_dataframe_input, 
              batch_start, batch_end, batch_len,
              langserver_url,
              progress=gr.Progress()):
        gr.Info('Batch processing started with Lang Server URL: {}'.format(langserver_url))

        if langserver_url:
            self.db_manager.append_message(LANGSERVE_URLS_TABLE, langserver_url)

        chain = RemoteRunnable(langserver_url)

        batch_message = _dataframe_to_batch_message(table_dataframe_input)
        messages = eval(batch_message)
        selected_messages = messages[int(batch_start):int(batch_end)]

        progress((0, len(selected_messages)), desc="Starting...")

        results = []
        for i in range(0, len(selected_messages), int(batch_len)):
            try:
                batch = selected_messages[i:i+int(batch_len)]
                result = chain.batch(batch)
                results.extend(result)
                progress((i+batch_len, len(selected_messages)), desc="Processing...")
            except Exception as e:
                raise gr.Error("Error: {} at batch {}".format(e, i))

        progress((len(selected_messages), len(selected_messages)), desc="Completed")

        # merge selected messages with results
        selected_messages_pd = pd.DataFrame(selected_messages)
        results_pd = pd.DataFrame(results, columns=['output'])
        df = pd.concat([selected_messages_pd, results_pd], axis=1)

        return gr.update(value=df)

    def update_input_files(self, df):
        # clear input if df is empty
        if df.shape[0] <= 1:
            return gr.update(value=None)

        filenames = _dump_dataframe(df, ['csv', 'xlsx'])

        return gr.update(value=filenames, visible=True)
    
    def update_output_files(self, df):
        # clear output if df is empty
        if df.shape[0] <= 1:
            return gr.update(value=None)

        file_types = ['csv', 'xlsx']
        filenames = _dump_dataframe(df, file_types)

        return gr.update(value=filenames, visible=True)
    
    def update_input_rows(self, df):
        df_size = len(df)
        return (
            gr.update(value=df_size),
            gr.update(value=0, minimum=0, maximum=df_size),
            gr.update(value=df_size, minimum=0, maximum=df_size),
            gr.update(value=df_size, minimum=0, maximum=df_size)
        )
    
def _dataframe_to_batch_message(table_dataframe_input):
    def process_row(row):
        return {
            col[1:] if col.startswith('!') else col: 
            eval(row[col]) if col.startswith('!') else row[col]
            for col in table_dataframe_input.columns
        }

    messages = [
        process_row(row) for _, row in table_dataframe_input.iterrows()
    ]
    return pprint.pformat(messages)

def _dump_dataframe(df, file_types):
    filenames = []
    for file_type in file_types:
        if file_type == 'csv':
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
                temp_filename = temp_file.name
                df.to_csv(temp_filename, index=False)
                filenames.append(temp_filename)
        elif file_type == 'xlsx':
            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_file:
                temp_filename = temp_file.name
                df.to_excel(temp_filename, index=False)
                filenames.append(temp_filename)
    return filenames