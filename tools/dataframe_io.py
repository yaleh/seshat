import tempfile

def dump_dataframe(df, file_types):
    # return None if df is None or empty, or the dim 0 is 1
    if df is None or df.empty or df.shape[0] <= 1:
        return None
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