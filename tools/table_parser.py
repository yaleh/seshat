import pandas as pd

class TableParser:
    @staticmethod
    def parse_markdown_table(md_content):
        df = None
        table_start_idx = md_content.find("| ")
        if table_start_idx >= 0:
            lines = md_content[table_start_idx:].split("\n")
            header_line = lines.pop(0).strip("| ").split(" | ")
            table_data = []
            for line in lines:
                if line.strip().startswith("|---"):
                    continue
                row = line.strip("| ").split(" | ")
                table_data.append(row)
            df = pd.DataFrame(table_data, columns=header_line)
        return df

    @staticmethod
    def add_skip_column(df):
        if 'Skip' not in df.columns:
            df.insert(0, 'Skip', '')
        return df