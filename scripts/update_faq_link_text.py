import pandas as pd
import secrets
import json
import sys

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
        answer = f"{output} 详见链接 [{question}](faq: {faq_id})"

    # return faq_id, json.dumps({"question": question, "answer": answer}, ensure_ascii=False)
    return faq_id, f"Q: {question}\nA: {answer}"

def process_faq_data(input_file, output_file):
    """Load data, process it by adding new columns, and save to a new Excel file."""
    df = pd.read_excel(input_file)
    df['faq_id'], df['faq_text'] = zip(*df.apply(construct_faq_text, axis=1))
    df.to_excel(output_file, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file.xlsx> <output_file.xlsx>")
        sys.exit(1)

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    process_faq_data(input_filename, output_filename)
