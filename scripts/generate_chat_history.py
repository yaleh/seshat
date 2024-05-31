import pandas as pd
import random
import sys

def generate_chat_history(data, num_arrays=5, num_rows=30):
    rows = []
    for _ in range(num_rows):
        chat_history = []
        for _ in range(num_arrays):
            idx = random.randint(0, len(data) - 1)
            chat_history.append([data.loc[idx, 'question'], data.loc[idx, 'output']])
        rows.append(str(chat_history))
    return pd.DataFrame(rows, columns=['!chat_history'])

def main(num_arrays, num_rows, input_file_path, output_file_path):
    # Load the Excel file
    data = pd.read_excel(input_file_path)

    # Generate the DataFrame with the new format
    new_data = generate_chat_history(data, num_arrays, num_rows)

    # Save the new DataFrame to the specified Excel file path
    new_data.to_excel(output_file_path, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py <num_arrays> <num_rows> <input_file_path> <output_file_path>")
    else:
        num_arrays = int(sys.argv[1])
        num_rows = int(sys.argv[2])
        input_file_path = sys.argv[3]
        output_file_path = sys.argv[4]
        main(num_arrays, num_rows, input_file_path, output_file_path)
