import gradio as gr
import pandas as pd
import tempfile
import secrets
import json
from bson import ObjectId

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

                    self.question_field = gr.Textbox(label="Question", value="Q")
                    self.summary_field = gr.Textbox(label="Summary", value="A")
                    self.q_and_a_field = gr.Textbox(label="Q&A", value="text")
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
                inputs=[self.data_before,self.question_field,self.summary_field,self.q_and_a_field],
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
        return str(ObjectId())

    def _construct_faq_text(self, row, question, summary, q_and_a):
        """Construct FAQ text for each row based on given rules."""
        question = row[question]
        output = row[summary]
        text = row[q_and_a]
        faq_id = self._generate_id()

        if output == '<NO ANSWER/>':
            answer = f"关于 `{question}`，详见链接 [{question}](faq:{faq_id})"
        else:
            answer = f"{output} 详见链接 [{question}](faq:{faq_id})"

        if text.startswith('Q: '):
            text = text.split('A: ')[1]

        return faq_id, f"Q: {question}\nA: {answer}", text

    def _process_faq_data(self, dataframe, question, summary, q_and_a):
        """Process the DataFrame to add new columns."""
        df = dataframe
        df['faq_id'], df['faq_text'], df['rdb_text'] = zip(*df.apply(self._construct_faq_text,
                                                 args=(question, summary, q_and_a),
                                                 axis=1))
        return df