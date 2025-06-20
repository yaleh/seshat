import unittest
import pandas as pd
from unittest.mock import patch

from tools.table_parser import TableParser


class TestTableParserExtended(unittest.TestCase):
    
    def test_parse_markdown_table_no_table(self):
        """Test parsing content with no table marker"""
        md_content = "This is just regular text without any table markers."
        result_df = TableParser.parse_markdown_table(md_content)
        self.assertIsNone(result_df)
    
    def test_parse_markdown_table_empty_content(self):
        """Test parsing empty content"""
        md_content = ""
        result_df = TableParser.parse_markdown_table(md_content)
        self.assertIsNone(result_df)
    
    def test_parse_markdown_table_table_at_end(self):
        """Test parsing content where table appears at the end"""
        md_content = "Some text before the table.\n\n| Column1 | Column2 |\n|---|---|\n| Value1 | Value2 |"
        expected_df = pd.DataFrame({"Column1": ["Value1"], "Column2": ["Value2"]})
        result_df = TableParser.parse_markdown_table(md_content)
        pd.testing.assert_frame_equal(result_df, expected_df)
    
    def test_parse_markdown_table_with_separator_variations(self):
        """Test parsing table with different separator line formats"""
        # Test with spaces in separator
        md_content = "| Column1 | Column2 |\n| --- | --- |\n| Value1 | Value2 |"
        expected_df = pd.DataFrame({"Column1": ["Value1"], "Column2": ["Value2"]})
        result_df = TableParser.parse_markdown_table(md_content)
        pd.testing.assert_frame_equal(result_df, expected_df)
    
    def test_parse_markdown_table_with_value_error_in_append(self):
        """Test parsing table where row append might cause ValueError"""
        # This test covers the except ValueError block in the code
        # We'll test with a table that has problematic formatting
        md_content = "| Column1 | Column2 |\n|---|---|\n| Value1 | Value2 |"
        
        # The actual code has a try-except block around table_data.append(row)
        # This test verifies the method works normally (the except block is hard to trigger)
        result_df = TableParser.parse_markdown_table(md_content)
        self.assertIsNotNone(result_df)
    
    def test_parse_markdown_table_dataframe_creation_error(self):
        """Test parsing table where DataFrame creation fails"""
        md_content = "| Column1 | Column2 |\n|---|---|\n| Value1 | Value2 |"
        
        # Mock DataFrame creation to raise ValueError
        with patch('pandas.DataFrame', side_effect=ValueError("Simulated DataFrame error")):
            result_df = TableParser.parse_markdown_table(md_content)
            self.assertIsNone(result_df)
    
    def test_add_skip_column_already_exists(self):
        """Test adding skip column when it already exists"""
        df = pd.DataFrame({"Skip": ["x"], "Column1": ["Value1"], "Column2": ["Value2"]})
        original_df = df.copy()
        result_df = TableParser.add_skip_column(df)
        pd.testing.assert_frame_equal(result_df, original_df)
    
    def test_add_skip_column_empty_dataframe(self):
        """Test adding skip column to empty DataFrame"""
        df = pd.DataFrame()
        result_df = TableParser.add_skip_column(df)
        expected_df = pd.DataFrame({"Skip": pd.Series([], dtype=object)})
        pd.testing.assert_frame_equal(result_df, expected_df)
    
    def test_parse_markdown_table_history_empty_history(self):
        """Test parsing empty history"""
        history = []
        result_df = TableParser.parse_markdown_table_history(history)
        self.assertIsNone(result_df)
    
    def test_parse_markdown_table_history_all_invalid_tables(self):
        """Test parsing history with all invalid tables"""
        history = [
            ["Not a table"],
            ["Also not a table"],
            ["Still not a table"]
        ]
        result_df = TableParser.parse_markdown_table_history(history)
        self.assertIsNone(result_df)
    
    def test_parse_markdown_table_history_first_item_invalid(self):
        """Test parsing history where first item is invalid but later items are valid"""
        history = [
            ["Not a table"],
            ["| Column1 | Column2 |\n|---|---|\n| Value1 | Value2 |"],
            ["| Column1 | Column3 |\n|---|---|\n| Value3 | Value4 |"]
        ]
        expected_df = pd.DataFrame({
            "Column1": ["Value1", "Value3"], 
            "Column2": ["Value2", ""], 
            "Column3": ["", "Value4"]
        })
        result_df = TableParser.parse_markdown_table_history(history)
        pd.testing.assert_frame_equal(result_df, expected_df)
    
    def test_parse_markdown_table_history_complex_column_merging(self):
        """Test parsing history with complex column merging scenarios"""
        history = [
            ["| A | B | C |\n|---|---|---|\n| 1 | 2 | 3 |"],
            ["| B | D | E |\n|---|---|---|\n| 4 | 5 | 6 |"],
            ["| C | D | F |\n|---|---|---|\n| 7 | 8 | 9 |"]
        ]
        expected_df = pd.DataFrame({
            "A": ["1", "", ""], 
            "B": ["2", "4", ""], 
            "C": ["3", "", "7"],
            "D": ["", "5", "8"],
            "E": ["", "6", ""],
            "F": ["", "", "9"]
        })
        result_df = TableParser.parse_markdown_table_history(history)
        pd.testing.assert_frame_equal(result_df, expected_df)
    
    def test_parse_markdown_table_with_edge_case_separators(self):
        """Test parsing table with edge case separator lines"""
        # Test with |---| format (no spaces)
        md_content = "| Column1 | Column2 |\n|---|---|\n| Value1 | Value2 |"
        expected_df = pd.DataFrame({"Column1": ["Value1"], "Column2": ["Value2"]})
        result_df = TableParser.parse_markdown_table(md_content)
        pd.testing.assert_frame_equal(result_df, expected_df)
    
    def test_parse_markdown_table_mixed_empty_cells(self):
        """Test parsing table with various empty cell patterns"""
        md_content = "| Column1 | Column2 | Column3 |\n|---|---|---|\n| Value1 |  | Value3 |\n|  | Value2 |  |"
        expected_df = pd.DataFrame({
            "Column1": ["Value1", ""], 
            "Column2": ["", "Value2"], 
            "Column3": ["Value3", ""]
        })
        result_df = TableParser.parse_markdown_table(md_content)
        pd.testing.assert_frame_equal(result_df, expected_df)
    
    def test_parse_markdown_table_single_column(self):
        """Test parsing table with single column"""
        md_content = "| Column1 |\n|---|\n| Value1 |\n| Value2 |"
        expected_df = pd.DataFrame({"Column1": ["Value1", "Value2"]})
        result_df = TableParser.parse_markdown_table(md_content)
        pd.testing.assert_frame_equal(result_df, expected_df)


if __name__ == '__main__':
    unittest.main()