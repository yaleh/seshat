
import unittest
import pandas as pd

from tools.table_parser import TableParser

class TestTableParser(unittest.TestCase):
    def test_parse_markdown_table(self):
        md_content = "| Column1 | Column2 |\n|---|---|\n| Value1 | Value2 |"
        expected_df = pd.DataFrame({"Column1": ["Value1"], "Column2": ["Value2"]})
        result_df = TableParser.parse_markdown_table(md_content)
        pd.testing.assert_frame_equal(result_df, expected_df)

        md_content = "| Column1 | Column2 | Column3 |\n|---|---|\n| Value1 | Value2 |"
        expected_df = pd.DataFrame({"Column1": ["Value1"], "Column2": ["Value2"], "Column3": [""]})
        result_df = TableParser.parse_markdown_table(md_content)
        pd.testing.assert_frame_equal(result_df, expected_df)        

    def test_add_skip_column(self):
        df = pd.DataFrame({"Column1": ["Value1"], "Column2": ["Value2"]})
        expected_df = pd.DataFrame({"Skip": [""], "Column1": ["Value1"], "Column2": ["Value2"]})
        result_df = TableParser.add_skip_column(df)
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_parse_markdown_table_history(self):
        history = [
            ["| Column1 | Column2 |\n|---|---|\n| Value1 | Value2 |"],
            ["| Column1 | Column3 |\n|---|---|\n| Value1 | Value3 |"]
        ]
        expected_df = pd.DataFrame({
            "Column1": ["Value1", "Value1"], 
            "Column2": ["Value2", ""], 
            "Column3": ["", "Value3"]
        })
        result_df = TableParser.parse_markdown_table_history(history)
        pd.testing.assert_frame_equal(result_df, expected_df)

        history = [
            ["| Column1 | Column2 |\n|---|---|\n| Value1 | Value2 |"],
            ["| Column1 | Column3 |\n|---|---|\n| Value1 | Value3 |"],
            ["| Column1 | Column2 |\n|---|---|\n| Value1 |"]
        ]
        expected_df = pd.DataFrame({
            "Column1": ["Value1", "Value1", "Value1"], 
            "Column2": ["Value2", "", ""], 
            "Column3": ["", "Value3", ""]
        })
        result_df = TableParser.parse_markdown_table_history(history)
        pd.testing.assert_frame_equal(result_df, expected_df)

if __name__ == '__main__':
    unittest.main()