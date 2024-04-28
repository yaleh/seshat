import gradio as gr
import pandas as pd
import tempfile
import secrets
import json

from tools.dataframe_io import dump_dataframe

class FAQSummaryFixUI:
    def __init__(self, config=None):
        self.config = config
        self.ui = self.init_ui()

    def init_ui(self):
        """Setup Gradio interface."""
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    self.data_after = gr.Dataframe()
                    self.download_output = gr.File(label="Download Processed File")
                with gr.Column():
                    self.data_before = gr.Dataframe()
                    with gr.Row():
                        self.file_input = gr.File(label="Upload your Excel file")
                        self.process_button = gr.Button("Process Data")

            self.file_input.change(
                lambda x: pd.read_excel(x.name),
                inputs=self.file_input,
                outputs=self.data_before
            )
            self.process_button.click(
                self._process_faq_data,
                inputs=self.data_before,
                outputs=[self.data_after]
            )
            self.data_after.change(
                lambda x: dump_dataframe(x, ['csv', 'xlsx']),
                inputs=self.data_after,
                outputs=self.download_output
            )

        return demo

    def _generate_id(self):
        """Generate a pseudo-Nano ID using hex format (16 bytes = 32 hex
        characters)."""
        return secrets.token_hex(16)

    def _construct_faq_text(self, row):
        """Construct FAQ text for each row based on given rules."""
        question = row['Q']
        output = row['Output']
        faq_id = self._generate_id()

        if output == '<NO ANSWER/>':
            answer = f"关于 {question}，详见链接 [{question}](faq: {faq_id})"
        else:
            answer = f"{output}\n详见链接 [{question}](faq: {faq_id})"

        return faq_id, json.dumps({"question": question, "answer": answer},
                                  ensure_ascii=False)

    def _process_faq_data(self, dataframe):
        """Process the DataFrame to add new columns."""
        df = dataframe
        df['faq_id'], df['faq_text'] = zip(*df.apply(self._construct_faq_text,
                                                     axis=1))
        return df