import tempfile

def dump_dataframe(df, file_types):
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