import pandas as pd

class TableParser:
    @staticmethod
    def parse_markdown_table(md_content):
        df = None
        table_start_idx = md_content.find("| ")
        if table_start_idx >= 0:
            lines = md_content[table_start_idx:].split("\n")
            # header_line = lines.pop(0).strip("| ").split(" | ")
            header_line = lines.pop(0).split("|")
            header_line = [cell.strip() for cell in header_line[1:-1]]
            table_data = []
            for line in lines:
                if line.strip().startswith("|---") or line.strip().startswith("| ---"):
                    continue

                # split line by "|", notice there might be empty cells like `| |`
                row = line.split("|")
                row = [cell.strip() for cell in row[1:-1]]

                # fix row if the number of columns doesn't match the header
                if len(row) < len(header_line):
                    row += [''] * (len(header_line) - len(row))
                elif len(row) > len(header_line):
                    row = row[:len(header_line)]
                    
                try:
                    table_data.append(row)
                except ValueError:
                    # skip invalid row
                    continue
            try:
                df = pd.DataFrame(table_data, columns=header_line)
            except ValueError:
                # skip invalid table
                pass
        return df

    @staticmethod
    def add_skip_column(df):
        if 'Skip' not in df.columns:
            df.insert(0, 'Skip', '')
        return df
    
    @staticmethod
    def parse_markdown_table_history(history):
        # history: list[list[str | tuple[str] | tuple[str | Path, str] | None]] | Callable | None
        df = None

        # parse items of history
        for item in history:
            item_df = TableParser.parse_markdown_table(item[-1])
            if item_df is None:
                continue
            if df is None:
                df = item_df
            else:
                # merge the columns of df and item_df
                # then append item_df to df

                # add the columns of item_df that are not in df
                for col in item_df.columns:
                    if col not in df.columns:
                        df[col] = ''
                # add the columns of df that are not in item_df
                for col in df.columns:
                    if col not in item_df.columns:
                        item_df[col] = ''
                # reorder the columns of item_df to match df
                item_df = item_df[df.columns]
                # append item_df to df
                df = pd.concat([df, item_df], ignore_index=True)

        return df



