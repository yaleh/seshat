import pandas as pd
import secrets
import json
import gradio as gr

def generate_id():
    """Generate a pseudo-Nano ID using hex format (16 bytes = 32 hex characters)."""
    return secrets.token_hex(16)

def construct_faq_text(row):
    """Construct FAQ text for each row based on given rules."""
    question = row['Q']
    output = row['Output']
    faq_id = generate_id()

    if output == '<NO ANSWER/>':
        answer = f"关于 {question}，详见链接 [{question}](faq: {faq_id})"
    else:
        answer = f"{output}\n详见链接 [{question}](faq: {faq_id})"

    return faq_id, json.dumps({"question": question, "answer": answer}, ensure_ascii=False)

def process_faq_data(dataframe):
    """Process the DataFrame to add new columns."""
    df = dataframe
    df['faq_id'], df['faq_text'] = zip(*df.apply(construct_faq_text, axis=1))
    return df


def setup_interface():
    """Setup Gradio interface."""
    with gr.Blocks() as demo:
        gr.Markdown("### FAQ Data Processing")
        with gr.Row():
            file_input = gr.File(label="Upload your Excel file")
            process_button = gr.Button("Process Data")
        data_before = gr.Dataframe()
        data_after = gr.Dataframe()
        download_output = gr.File(label="Download Processed File")

        file_input.change(lambda x: pd.read_excel(x.name), inputs=file_input, outputs=data_before)
        process_button.click(process_faq_data, inputs=data_before, outputs=[data_after])

    return demo

app = setup_interface()
app.launch()
