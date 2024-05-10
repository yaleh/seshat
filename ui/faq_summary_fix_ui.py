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

                with gr.Column():
                    self.data_before = gr.Dataframe()
                    with gr.Row():
                        self.file_input = gr.File(label="Upload your Excel file")
                        self.process_button = gr.Button("Process Data")
            
            with gr.Row():
                with gr.Column():
                    self.question_field = gr.Textbox(
                        label="Question Field", value="Q")
                    self.summary_field = gr.Textbox(
                        label="Summary Field", value="A")
                    self.q_and_a_field = gr.Textbox(
                        label="Q&A Field", value="text")

                with gr.Column():
                    self.faq_id_output_field = gr.Textbox(
                        label="FAQ ID Output Field", value="faq_id")
                    self.faq_text_output_field = gr.Textbox(
                        label="FAQ Text Output Field", value="faq_text")
                    self.rdb_text_output_field = gr.Textbox(
                        label="RDB Text Output Field", value="rdb_text")

                with gr.Column():
                    self.no_answer_pattern = gr.Textbox(
                        label="No Answer Pattern", value="<NO ANSWER/>")
                    self.no_answer_output_template = gr.Textbox(
                        label="No Answer Output Template", value="For more information about `{question}`, please visit the following link: [{question}](faq:{faq_id}).")
                    self.general_output_template = gr.Textbox(
                        label="General Output Template", value="{output} For more details, please see the link [{question}](faq:{faq_id}).")

            self.file_input.change(
                self._excel_file_uploaded,
                inputs=self.file_input,
                outputs=self.data_before
            )
            self.process_button.click(
                self._process_faq_data,
                inputs=[self.data_before,
                        self.question_field, self.summary_field, self.q_and_a_field,
                        self.faq_id_output_field, self.faq_text_output_field, self.rdb_text_output_field,
                        self.no_answer_pattern, self.no_answer_output_template, self.general_output_template
                        ],
                outputs=[self.data_after]
            )
            self.data_after.change(
                lambda x: dump_dataframe(x, ['csv', 'xlsx']),
                inputs=self.data_after,
                outputs=self.download_output
            )

        return demo
    
    def _excel_file_uploaded(self, file):
        if file is None:
            return None
        return pd.read_excel(file.name)

    def _generate_id(self):
        """Generate a pseudo-Nano ID using hex format (16 bytes = 32 hex
        characters)."""
        return str(ObjectId())

    def _construct_faq_text(self, row, 
                            question_field, summary_field, q_and_a_field,
                            no_answer_pattern, no_answer_output_template, general_output_template
                            ):
        """Construct FAQ text for each row based on given rules."""
        question_field = row[question_field]
        output = row[summary_field]
        text = row[q_and_a_field]
        faq_id = self._generate_id()

        if output == no_answer_pattern:
            answer = no_answer_output_template.format(question=question_field, faq_id=faq_id, output=output, text=text)
        else:
            answer = general_output_template.format(question=question_field, faq_id=faq_id, output=output, text=text)

        if text.startswith('Q: '):
            text = text.split('A: ')[1]

        return faq_id, f"Q: {question_field}\nA: {answer}", text

    def _process_faq_data(self, dataframe, question_field, summary_field, q_and_a_field,
                          faq_id_output_field, faq_text_output_field, rdb_text_output_field,
                          no_answer_pattern, no_answer_output_template, general_output_template):
        """Process the DataFrame to add new columns."""
        try:
            df = dataframe
            df[faq_id_output_field], df[faq_text_output_field], df[rdb_text_output_field] = \
                zip(*df.apply(
                    self._construct_faq_text,
                    args=(
                        question_field, 
                        summary_field, 
                        q_and_a_field,
                        no_answer_pattern, 
                        no_answer_output_template, 
                        general_output_template
                    ),
                    axis=1
                ))
            return df
        except Exception as e:
            raise gr.Error(f"Error processing data: {e}")