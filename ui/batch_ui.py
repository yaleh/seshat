import gradio as gr
import pandas as pd
import tempfile

from db.db_sqlite3 import DatabaseManager
from components.lcel import LLMModelFactory
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.schema.output_parser import StrOutputParser
from tools.table_parser import TableParser
from tools.utils import detect_encoding

class BatchUI:
    # def __init__(self, db_manager, llmbot):
    def __init__(self, config):

        self.config = config
        self.default_model_service = self.config.llm.default_model_service
        self.model_type = self.config.llm.llm_services[self.default_model_service].type
        self.model_name = self.config.llm.llm_services[
            self.default_model_service].default_model

        self.model_args = self.config.llm.llm_services[self.default_model_service].args

        self.db_manager = DatabaseManager(
            self.config.server.message_db, 
            self.config.server.max_message_length
        )
        self.model_factory = LLMModelFactory()

        self.ui = self.init_ui()

    def init_ui(self):
        with gr.Blocks() as batch_ui:
            with gr.Row():
                with gr.Column():
                    self.chatbot = gr.Chatbot(show_share_button=True, show_copy_button=True)

                    with gr.Blocks():
                        with gr.Tab("Manual Collecting"):
                            with gr.Group():
                                with gr.Row():
                                    self.updating_table_method = gr.Dropdown(
                                        choices=["Markdown Table", "Chat History", "History w/ Tables"],
                                        value="Markdown Table",
                                        show_label=False,
                                        allow_custom_value=False
                                    )
                                    self.update_table_button = gr.Button(value="Get Table from Message")
                                self.table_dataframe_output = gr.Dataframe(
                                    label="Table Dataframe for Output",
                                    value=[],
                                    wrap=True,
                                    interactive=True
                                )
                            with gr.Group():
                                self.merge_table_button = gr.Button(value="Merge Table")
                                self.table_dataframe_merged = gr.Dataframe(
                                    label="Merged Table Dataframe",
                                    value=[],
                                    wrap=True,
                                    interactive=True
                                )
                                with gr.Row():
                                    self.download_file_merged = gr.File(
                                        label='Download Merged Result File',
                                        file_count='single',
                                        file_types=['csv', 'xls', 'xlsx'],
                                        visible=False
                                    )
                                    self.file_type2 = gr.Dropdown(
                                        choices=['csv', 'xls', 'xlsx'],
                                        value='csv',
                                        label='Download filetype'
                                    )
                                    self.download_df_merged = gr.Button(
                                        value='Save Merged Dataframe as File'
                                    )

                        with gr.Tab("Auto Collecting"):

                            with gr.Group():
                                self.table_dataframe_result = gr.Dataframe(
                                    label="Batch result as Dataframe Column",
                                    value=[],
                                    wrap=True,
                                    interactive=True
                                )
                                with gr.Row():
                                    self.download_file_res = gr.File(
                                        label='Download Result File',
                                        file_count='single',
                                        file_types=['csv', 'xls', 'xlsx'],
                                        visible=False
                                    )
                                    self.file_type = gr.Dropdown(
                                        choices=['csv', 'xls', 'xlsx'],
                                        value='csv',
                                        label='Download filetype'
                                    )
                                    self.download_df_res = gr.Button(
                                        value='Save Dataframe Results as File'
                                    )

                with gr.Column():
                    with gr.Group():
                        with gr.Row():
                            self.llm_service_dropdown = gr.Dropdown(
                                label='Choice LLM Service',
                                choices=self.config.llm.llm_services.keys(),
                                value=self.default_model_service,
                                interactive=True,
                                allow_custom_value=False
                            )
                            self.llm_model_name_dropdown = gr.Dropdown(
                                label="Choice LLM Model Name",
                                choices=self.config.llm.llm_services[self.default_model_service].models,
                                value=self.model_name,
                                interactive=True,
                                allow_custom_value=True
                            )
                        self.llm_other_option_checkbox = gr.Checkbox(value=False,
                                                                    label="Other LLM Model Options"
                                                                    )
                        self.llm_model_temperature_slider = gr.Slider(visible=False,
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=0.0,
                            interactive=True,
                            label="LLM Model Temperature"
                        )
                        self.llm_model_max_tokens_slider = gr.Slider(visible=False,
                            minimum=0,
                            maximum=32000,
                            step=256,
                            value=0,
                            interactive=True,
                            label="LLM Model Token Limit (0 for auto)"
                        )
                        self.llm_model_request_timeout_slider = gr.Slider(visible=False,
                            minimum=0,
                            maximum=600,
                            step=5,
                            value=600,
                            interactive=True,
                            label="LLM Model Timeout"
                        )
                        self.llm_model_max_retries_slider = gr.Slider(visible=False,
                            minimum=0,
                            maximum=30,
                            step=1,
                            value=6,
                            interactive=True,
                            label="LLM Model Max Retries"
                        )

                    self.system_prompt_tab = gr.Tab('System and User Prompt')
                    self.string_prompt_tab = gr.Tab('String Prompt')

                    with self.system_prompt_tab:
                        self.system_msg_textbox = gr.Textbox(label="System Prompt", lines=5)
                        self.system_msg = gr.Dropdown(
                            choices=self.db_manager.get_messages("system_messages"), 
                            label="System Prompt History", allow_custom_value=True
                        )
                        self.user_msg_textbox = gr.Textbox(label="User Prompt", lines=5)
                        self.user_msg = gr.Dropdown(
                            choices=self.db_manager.get_messages("user_messages"), 
                            label="User Prompt History", allow_custom_value=True
                        )
                    with self.string_prompt_tab:
                        self.string_msg_textbox = gr.Textbox(label="String Prompt", lines=5)
                        self.string_msg = gr.Dropdown(
                            choices=self.db_manager.get_messages("string_messages"), 
                            label="String Prompt History", allow_custom_value=True
                        )
                    # self.csv_output_checkbox = gr.Checkbox(label="Batch Process By Table", value=True)

                    with gr.Group():
                        with gr.Tab("Single"):
                            self.send_call = gr.Button(value="Send(call)", variant='primary')

                        with gr.Tab("Batch"):
                            self.table_dataframe_input = gr.Dataframe(
                                label="Table Dataframe for Input",
                                value=[],
                                wrap=True,
                                interactive=True
                            )
                            with gr.Row():
                                self.table_rows = gr.Number(label="Table Rows", value=0, precision=0, 
                                                            interactive=False)
                                self.refresh_table_rows = gr.Button(value="Refresh Table Rows")
                            with gr.Row():
                                self.table_file = gr.File(
                                    label="Upload file(*.csv/xls/xlsx)",
                                    file_types=["csv", "xls", "xlsx"],
                                    height=120
                                )
                                with gr.Column():
                                    self.batch_start = gr.Number(label="Batch Start", value=0, minimum=0, precision=0)
                                    self.batch_end = gr.Number(label="Batch End (excluded)", value=0, minimum=0, precision=0)
                                    self.batch_size = gr.Number(label="Batch Size", value=10, minimum=0, precision=0)
                            with gr.Row():
                                self.batch_send_button = gr.Button(value="Batch Send", variant='primary')
                                self.cancel_batch_button = gr.Button(value="Cancel Batch Send")

                        with gr.Group():
                            with gr.Row():
                                self.clear_output = gr.ClearButton([self.chatbot],
                                                                value="Clear Output")
                                self.clear_all = gr.ClearButton([self.user_msg, self.system_msg,
                                                                self.chatbot], value="Clear All")
                    
            # Bind events
            self.bind_events()

        return batch_ui

    def bind_events(self):
        self.llm_other_option_checkbox.change(
            self.show_other_llm_option,
            inputs=[self.llm_other_option_checkbox],
            outputs=[self.llm_model_temperature_slider, self.llm_model_max_tokens_slider,
                     self.llm_model_request_timeout_slider, self.llm_model_max_retries_slider]
        )
        self.llm_service_dropdown.change(
            self.update_llm_model_name_dropdown,
            [self.llm_service_dropdown],
            [self.llm_model_name_dropdown]
        )
        self.llm_model_name_dropdown.change(
            fn=self.update_llm_config,
            inputs=[self.llm_service_dropdown, self.llm_model_name_dropdown]
        )
        self.user_msg.change(
            lambda s: s,
            [self.user_msg],
            [self.user_msg_textbox]
        )
        self.system_msg.change(
            lambda s: s,
            [self.system_msg],
            [self.system_msg_textbox]
        )
        self.chatbot.change(
            self.update_system_msg_and_user_msg,
            [],
            [self.system_msg, self.user_msg]
        )
        self.send_call.click(
            self.send_call_func,
            [self.system_msg_textbox, self.user_msg_textbox, self.chatbot],
            [self.chatbot]
        )
        self.system_msg.input(
            lambda s: self.update_dropdown_choices("system_messages", s),
            [self.system_msg],
            [self.system_msg]
        )
        self.user_msg.input(
            lambda s: self.update_dropdown_choices("user_messages", s),
            [self.user_msg],
            [self.user_msg]
        )

        # self.csv_output_checkbox.change(
        #     lambda checked: (gr.update(visible=checked), gr.update(visible=not checked)),
        #     [self.csv_output_checkbox],
        #     [table_block, self.send_call]
        # )
        self.update_table_button.click(
            self.update_table,
            [self.chatbot, self.updating_table_method],
            [self.table_dataframe_output, self.table_file]
        )
        self.table_file.upload(
            self.upload_table,
            [self.table_file],
            [self.table_dataframe_input, self.batch_start, self.batch_end, self.batch_size]
        )

        self.refresh_table_rows.click(
            self.get_table_rows,
            [self.table_dataframe_input],
            [self.table_rows]
        )

        batch_send_env = self.batch_send_button.click(
            self.send_batch_func,
            [self.system_msg_textbox, self.user_msg_textbox, self.chatbot,
             self.table_dataframe_input, self.batch_start, self.batch_end, self.batch_size],
            [self.chatbot, self.table_dataframe_result]
        )

        self.cancel_batch_button.click(
            self.cancel_batch_send,
            cancels=[batch_send_env]
        )
        self.merge_table_button.click(
            fn=self.merge_table,
            inputs=[self.table_dataframe_input, self.table_dataframe_output,
                self.batch_start, self.batch_end],
            outputs=[self.table_dataframe_merged]
        )
        self.download_df_res.click(
            fn=self.save_dataframe,
            inputs=[self.file_type, self.table_dataframe_result],
            outputs=[self.download_file_res]
        )
        self.download_df_merged.click(
            fn=self.save_dataframe,
            inputs=[self.file_type2, self.table_dataframe_merged],
            outputs=[self.download_file_merged]
        )

    def merge_table(self, table_input, table_output, batch_start, batch_end):
        try:
            selected_table = table_input.iloc[int(batch_start):int(batch_end), :]
            refined_table_input = selected_table.loc[table_input['Skip']=='', :]
            refined_table_input.reset_index(drop=True, inplace=True)
        except Exception as e:
            raise gr.Error('请确保已上传 `Table Dataframe for Input`: %s' % e)
        result = pd.concat([refined_table_input, table_output], axis=1, ignore_index=False)
        return result

    def update_llm_config(self, llm_service, llm_model_name):
        model_service = self.config.llm.llm_services[llm_service]
        self.model_args = model_service.args
        self.model_type = model_service.type

        self.model_name = llm_model_name

    def show_other_llm_option(self, visible_flag):
        slider_list = []
        if 'temperature' in list(self.model_args.keys()):
            slider_list.append(
                gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0,
                          interactive=True, label="LLM Model Temperature",
                          visible=visible_flag)
            )
        else:
            slider_list.append(
                gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0,
                          interactive=True, label="LLM Model Temperature",
                          visible=False)
            )
        if ('max_tokens' in list(self.model_args.keys()) or
                'max_length' in list(self.model_args.keys())):
            slider_list.append(
                gr.Slider(minimum=0, maximum=32000, step=256, value=0,
                          interactive=True, label="LLM Model Token Limit (0 for auto)",
                          visible=visible_flag)
            )
        else:
            slider_list.append(
                gr.Slider(minimum=0, maximum=32000, step=256, value=0,
                          interactive=True, label="LLM Model Token Limit (0 for auto)",
                          visible=False)
            )
        if 'request_timeout' in list(self.model_args.keys()):
            slider_list.append(
                gr.Slider(minimum=0, maximum=600, step=5, value=600,
                          interactive=True, label="LLM Model Request Timeout",
                          visible=visible_flag)
            )
        else:
            slider_list.append(
                gr.Slider(minimum=0, maximum=600, step=5, value=600,
                          interactive=True, label="LLM Model Request Timeout",
                          visible=False)
            )
        if 'max_retries' in list(self.model_args.keys()):
            slider_list.append(
                gr.Slider(minimum=0, maximum=30, step=1, value=6,
                          interactive=True, label="LLM Model Max Retries",
                          visible=visible_flag)
            )
        else:
            slider_list.append(
                gr.Slider(minimum=0, maximum=30, step=1, value=6,
                          interactive=True, label="LLM Model Max Retries",
                          visible=False)
            )
        return slider_list

    def update_llm_model_name_dropdown(self, cur_llm_service):
        # self.model_name_list = self.config_loader.get_model_name_list(cur_llm_service)
        # self.model_name_list = self.config.llm.llm_services[cur_llm_service].models
        self.default_model_service = cur_llm_service
        return gr.Dropdown(choices=self.config.llm.llm_services[self.default_model_service].models)

    def update_system_msg_and_user_msg(self):
        system_msg = gr.update(choices=self.db_manager.get_messages("system_messages"))
        user_msg = gr.update(choices=self.db_manager.get_messages("user_messages"))
        return system_msg, user_msg

    def __create_chat_prompt(self, system_prompt, user_prompt, chat_history):
        # 将system_message 和 user_message 存入数据库
        if isinstance(user_prompt, str):
            self.db_manager.append_message("user_messages", user_prompt)
        if isinstance(system_prompt, str):
            self.db_manager.append_message("system_messages", system_prompt)
        # prompt
        chat_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
        for human, ai in chat_history:
            chat_prompt += HumanMessage(content=human)
            chat_prompt += AIMessage(content=ai)
        chat_prompt += HumanMessagePromptTemplate.from_template(user_prompt)
        return chat_prompt

    def create_chat_chain(self, chat_prompt):
        output_parser = StrOutputParser()
        llmbot = self.model_factory.create_model(model_type=self.model_type, 
                                                 model_name=self.model_name,
                                                 **self.model_args.dict())
        chat_chain = chat_prompt | llmbot | output_parser
        return chat_chain

    def send_call_func(self, system_prompt, user_prompt, chat_history):
        gr.Info(f'model service: {self.default_model_service}')
        gr.Info(f'model name: {self.model_name}')

        try:
            if isinstance(user_prompt, str):
                self.db_manager.append_message("user_messages", user_prompt)
            if isinstance(system_prompt, str):
                self.db_manager.append_message("system_messages", system_prompt)

            chat_chain = self.model_factory.create_model(
                model_type=self.model_type, 
                model_name=self.model_name,
                **self.model_args.dict()
            ) | StrOutputParser()
            llmbot_res = chat_chain.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            chat_history.append((user_prompt, llmbot_res))
        except Exception as e:
            raise gr.Error(f"send_call_func-err:{e}")
        return gr.Chatbot(value=chat_history)

    def send_batch_func(self, system_prompt, user_prompt, chat_history, table,
                        batch_start, batch_end, batch_len, progress=gr.Progress()):
        gr.Info(f'model service: {self.default_model_service}')
        gr.Info(f'model name: {self.model_name}')

        chat_prompt = self.__create_chat_prompt(system_prompt, user_prompt, chat_history)
        chat_chain = self.create_chat_chain(chat_prompt)

        selected_table = table.iloc[int(batch_start):int(batch_end), :]
        table_refined = selected_table.loc[selected_table['Skip']=='', :] \
            if 'Skip' in selected_table.columns else selected_table

        table_output = table_refined.copy()
        table_output.insert(table_output.shape[-1], 'Output', 0)

        progress((0, len(table_refined)), desc="Starting...")
        
        # 根据batch step，对table进行切片处理
        for i in range(0,len(table_refined), batch_len):
            try:
                table_step = table_refined.iloc[i:i+batch_len,:]
                # 构造chain需要的输入
                input_list = []
                for _, row in table_step.iterrows():
                    kv_item = {}
                    for input_varivable in chat_prompt.input_variables:
                        kv_item[input_varivable] = row[input_varivable]
                    input_list.append(kv_item)

                try:
                    llmbot_res = chat_chain.batch(input_list)
                except Exception as e:
                    # stop the batch process if self.config.batch has attr 'stop_on_error' and it is True
                    if hasattr(self.config.batch, 'stop_on_error') and self.config.batch.stop_on_error:
                        raise gr.Error(f'send_batch_func-err:{e} at batch {i} to {i+batch_len}')
                    gr.Warning(f'send_batch_func-err:{e} at batch {i} to {i+batch_len}')

                    # fallback to a loop of invoke()
                    llmbot_res = [''] * len(input_list)
                    for j in range(len(input_list)):
                        try:
                            llmbot_res[j] = chat_chain.invoke(input_list[j])
                        except Exception as e:
                            gr.Warning(f'send_batch_func-err:{e} at item {i+j}')
                            llmbot_res[j] = ''
                human_list = chat_prompt.batch(input_list)
                human_list = [item.messages[1].content for item in human_list]
                chat_history += list(zip(human_list, llmbot_res))
                # table_output.iloc[i:i+batch_len,table_output.shape[-1]-1]=llmbot_res
                target_series = table_output.iloc[i:i+batch_len,table_output.shape[-1]-1].astype(object)
                target_series.update(pd.Series(llmbot_res))
                gr.Info(f"当前已完成/总条数：{i+batch_len if (i+batch_len) < len(selected_table) else len(selected_table)}/"
                        f"{len(selected_table)}")
                progress((i+batch_len, len(table_refined)), desc="Processing...")
            except Exception as e:
                # if e is an gr.Error, raise it directly
                if isinstance(e, gr.Error):
                    raise e
                else:
                    raise gr.Error(f'send_batch_func-err:{e} at batch {i} to {i+batch_len}')

        return gr.Chatbot(value=chat_history), table_output

    def cancel_batch_send(self):
        # batch_send_env.cancel()
        pass

    def update_dropdown_choices(self, table, string):
        messages = self.db_manager.get_messages(table)
        if not string or string.strip() == "":
            return gr.update(choices=messages)
        matching_messages = [message for message in messages if string.lower() in message.lower()]
        if len(matching_messages) == 1 and matching_messages[0] == string:
            return gr.update(choices=messages)
        return gr.update(choices=matching_messages)

    def update_table(self, chatbot, updating_table_method):
        if updating_table_method == "Markdown Table":
            try:
                data = TableParser.add_skip_column(TableParser.parse_markdown_table(chatbot[-1][-1]))
                with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
                    temp_filename = temp_file.name
                    data.to_csv(temp_filename, index=False)
                    return data, gr.File(value=temp_filename, visible=True)
            except (IndexError, TypeError):
                return None, None
        elif updating_table_method == "Chat History":
            try:
                data = pd.DataFrame(chatbot, columns=["Input", "Output"])
                with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
                    temp_filename = temp_file.name
                    data.to_csv(temp_filename, index=False)
                    return data, gr.File(value=temp_filename, visible=True)
            except Exception as e:
                raise gr.Error(f"save_dataframe-err:{e}")
        elif updating_table_method == "History w/ Tables":
            try:
                data = TableParser.parse_markdown_table_history(chatbot)
                with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
                    temp_filename = temp_file.name
                    data.to_csv(temp_filename, index=False)
                    return data, gr.File(value=temp_filename, visible=True)
            except Exception as e:
                raise gr.Error(f"save_dataframe-err:{e}")
        else:
            return None, None

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
        df = TableParser.add_skip_column(df)
        df_size = len(df)
        return (
            df,
            gr.update(value=0, minimum=0, maximum=df_size),
            gr.update(value=df_size, minimum=0, maximum=df_size),
            gr.update(value=df_size, minimum=0, maximum=df_size)
        )
    def get_table_rows(self, table):
        return table.shape[0]

    def save_dataframe(self, file_type, df):
        if file_type == 'csv':
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
                temp_filename = temp_file.name
                df.to_csv(temp_filename, index=False)
        elif file_type == 'xls' or file_type == 'xlsx':
            with tempfile.NamedTemporaryFile(suffix=f".{file_type}", delete=False) as temp_file:
                temp_filename = temp_file.name
                df.to_excel(temp_filename,sheet_name="sheet1",index=False)

        return gr.update(value=temp_filename, visible=True)
