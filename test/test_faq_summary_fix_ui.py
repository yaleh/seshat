import unittest
import pandas as pd
import tempfile
from unittest.mock import patch, MagicMock, Mock
import gradio as gr

from ui.faq_summary_fix_ui import FAQSummaryFixUI


class TestFAQSummaryFixUIBasics(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
    
    @patch('ui.faq_summary_fix_ui.FAQSummaryFixUI.init_ui')
    def test_faq_summary_fix_ui_init_basics(self, mock_init_ui):
        """Test FAQSummaryFixUI basic initialization"""
        mock_ui_instance = MagicMock()
        mock_init_ui.return_value = mock_ui_instance
        
        # Create FAQSummaryFixUI
        faq_ui = FAQSummaryFixUI(self.mock_config)
        
        # Verify initialization
        self.assertEqual(faq_ui.config, self.mock_config)
        self.assertEqual(faq_ui.ui, mock_ui_instance)
        
        # Verify UI was created
        mock_init_ui.assert_called_once()
    
    @patch('ui.faq_summary_fix_ui.FAQSummaryFixUI.init_ui')
    def test_faq_summary_fix_ui_init_with_none_config(self, mock_init_ui):
        """Test FAQSummaryFixUI initialization with None config"""
        mock_ui_instance = MagicMock()
        mock_init_ui.return_value = mock_ui_instance
        
        # Create FAQSummaryFixUI with None config
        faq_ui = FAQSummaryFixUI(None)
        
        # Verify initialization
        self.assertIsNone(faq_ui.config)
        self.assertEqual(faq_ui.ui, mock_ui_instance)


class TestFAQSummaryFixUIComponents(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
    
    @patch('ui.faq_summary_fix_ui.dump_dataframe')
    @patch('ui.faq_summary_fix_ui.gr.Blocks')
    def test_init_ui_creates_components(self, mock_blocks, mock_dump_dataframe):
        """Test init_ui creates all required UI components"""
        mock_ui = MagicMock()
        mock_blocks.return_value.__enter__.return_value = mock_ui
        
        with patch('ui.faq_summary_fix_ui.gr.Row'), \
             patch('ui.faq_summary_fix_ui.gr.Column'), \
             patch('ui.faq_summary_fix_ui.gr.File') as mock_file, \
             patch('ui.faq_summary_fix_ui.gr.Button') as mock_button, \
             patch('ui.faq_summary_fix_ui.gr.Dataframe') as mock_dataframe, \
             patch('ui.faq_summary_fix_ui.gr.Accordion'), \
             patch('ui.faq_summary_fix_ui.gr.Textbox') as mock_textbox:
            
            # Setup mocks for UI components
            mock_file_instances = [MagicMock() for _ in range(2)]
            mock_file.side_effect = mock_file_instances
            
            mock_button_instance = MagicMock()
            mock_button.return_value = mock_button_instance
            
            mock_dataframe_instances = [MagicMock() for _ in range(2)]
            mock_dataframe.side_effect = mock_dataframe_instances
            
            mock_textbox_instances = [MagicMock() for _ in range(9)]
            mock_textbox.side_effect = mock_textbox_instances
            
            faq_ui = FAQSummaryFixUI(self.mock_config)
            
            # Verify components are stored
            self.assertTrue(hasattr(faq_ui, 'file_input'))
            self.assertTrue(hasattr(faq_ui, 'download_output'))
            self.assertTrue(hasattr(faq_ui, 'process_button'))
            self.assertTrue(hasattr(faq_ui, 'data_before'))
            self.assertTrue(hasattr(faq_ui, 'data_after'))
            
            # Verify textbox components for parameters
            self.assertTrue(hasattr(faq_ui, 'question_field'))
            self.assertTrue(hasattr(faq_ui, 'summary_field'))
            self.assertTrue(hasattr(faq_ui, 'q_and_a_field'))
            self.assertTrue(hasattr(faq_ui, 'faq_id_output_field'))
            self.assertTrue(hasattr(faq_ui, 'faq_text_output_field'))
            self.assertTrue(hasattr(faq_ui, 'rdb_text_output_field'))
            self.assertTrue(hasattr(faq_ui, 'no_answer_pattern'))
            self.assertTrue(hasattr(faq_ui, 'no_answer_output_template'))
            self.assertTrue(hasattr(faq_ui, 'general_output_template'))
            
            # Verify event bindings were set up
            faq_ui.file_input.change.assert_called_once()
            faq_ui.process_button.click.assert_called_once()
            faq_ui.data_after.change.assert_called_once()
    
    @patch('ui.faq_summary_fix_ui.dump_dataframe')
    @patch('ui.faq_summary_fix_ui.gr.Blocks')
    def test_init_ui_with_default_values(self, mock_blocks, mock_dump_dataframe):
        """Test init_ui sets correct default values for textboxes"""
        mock_ui = MagicMock()
        mock_blocks.return_value.__enter__.return_value = mock_ui
        
        with patch('ui.faq_summary_fix_ui.gr.Row'), \
             patch('ui.faq_summary_fix_ui.gr.Column'), \
             patch('ui.faq_summary_fix_ui.gr.File'), \
             patch('ui.faq_summary_fix_ui.gr.Button'), \
             patch('ui.faq_summary_fix_ui.gr.Dataframe'), \
             patch('ui.faq_summary_fix_ui.gr.Accordion'), \
             patch('ui.faq_summary_fix_ui.gr.Textbox') as mock_textbox:
            
            faq_ui = FAQSummaryFixUI(self.mock_config)
            
            # Check that textboxes were created with correct default values
            textbox_calls = mock_textbox.call_args_list
            
            # Verify specific default values
            question_field_call = next((call for call in textbox_calls 
                                       if 'label' in call.kwargs and call.kwargs['label'] == "Question Field"), None)
            self.assertIsNotNone(question_field_call)
            self.assertEqual(question_field_call.kwargs['value'], "Q")
            
            summary_field_call = next((call for call in textbox_calls 
                                      if 'label' in call.kwargs and call.kwargs['label'] == "Summary Field"), None)
            self.assertIsNotNone(summary_field_call)
            self.assertEqual(summary_field_call.kwargs['value'], "A")
            
            q_and_a_field_call = next((call for call in textbox_calls 
                                      if 'label' in call.kwargs and call.kwargs['label'] == "Q&A Field"), None)
            self.assertIsNotNone(q_and_a_field_call)
            self.assertEqual(q_and_a_field_call.kwargs['value'], "text")


class TestFAQSummaryFixUIFileOperations(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
    
    @patch('ui.faq_summary_fix_ui.FAQSummaryFixUI.init_ui')
    def test_excel_file_uploaded_success(self, mock_init_ui):
        """Test _excel_file_uploaded with valid Excel file"""
        mock_init_ui.return_value = MagicMock()
        
        faq_ui = FAQSummaryFixUI(self.mock_config)
        
        # Create mock file
        mock_file = MagicMock()
        mock_file.name = 'test.xlsx'
        
        test_df = pd.DataFrame({'Q': ['Question 1', 'Question 2'], 'A': ['Answer 1', 'Answer 2']})
        
        with patch('pandas.read_excel') as mock_read_excel:
            mock_read_excel.return_value = test_df
            
            result = faq_ui._excel_file_uploaded(mock_file)
            
            # Verify file reading
            mock_read_excel.assert_called_once_with('test.xlsx')
            
            # Verify result
            pd.testing.assert_frame_equal(result, test_df)
    
    @patch('ui.faq_summary_fix_ui.FAQSummaryFixUI.init_ui')
    def test_excel_file_uploaded_none_file(self, mock_init_ui):
        """Test _excel_file_uploaded with None file"""
        mock_init_ui.return_value = MagicMock()
        
        faq_ui = FAQSummaryFixUI(self.mock_config)
        
        result = faq_ui._excel_file_uploaded(None)
        
        # Should return None for None input
        self.assertIsNone(result)


class TestFAQSummaryFixUIIDGeneration(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
    
    @patch('ui.faq_summary_fix_ui.FAQSummaryFixUI.init_ui')
    @patch('ui.faq_summary_fix_ui.ObjectId')
    def test_generate_id(self, mock_object_id, mock_init_ui):
        """Test _generate_id method"""
        mock_init_ui.return_value = MagicMock()
        mock_object_id.return_value = 'mock_object_id_123'
        
        faq_ui = FAQSummaryFixUI(self.mock_config)
        
        result = faq_ui._generate_id()
        
        # Verify ObjectId was called
        mock_object_id.assert_called_once()
        
        # Verify result is string representation
        self.assertEqual(result, 'mock_object_id_123')


class TestFAQSummaryFixUITextConstruction(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
    
    @patch('ui.faq_summary_fix_ui.FAQSummaryFixUI.init_ui')
    def test_construct_faq_text_with_no_answer_pattern(self, mock_init_ui):
        """Test _construct_faq_text with no answer pattern"""
        mock_init_ui.return_value = MagicMock()
        
        faq_ui = FAQSummaryFixUI(self.mock_config)
        
        # Mock the _generate_id method
        with patch.object(faq_ui, '_generate_id', return_value='test_id_123'):
            # Create test row
            row = pd.Series({
                'Q': 'What is the weather?',
                'A': '<NO ANSWER/>',
                'text': 'Q: What is the weather?\nA: No information available'
            })
            
            result_id, result_faq_text, result_rdb_text = faq_ui._construct_faq_text(
                row, 'Q', 'A', 'text',
                '<NO ANSWER/>', 
                'For more information about `{question}`, please visit the following link: [{question}](faq:{faq_id}).',
                '{output} For more details, please see the link [{question}](faq:{faq_id}).'
            )
            
            # Verify results
            self.assertEqual(result_id, 'test_id_123')
            expected_faq_text = 'Q: What is the weather?\nA: For more information about `What is the weather?`, please visit the following link: [What is the weather?](faq:test_id_123).'
            self.assertEqual(result_faq_text, expected_faq_text)
            self.assertEqual(result_rdb_text, 'No information available')
    
    @patch('ui.faq_summary_fix_ui.FAQSummaryFixUI.init_ui')
    def test_construct_faq_text_with_general_answer(self, mock_init_ui):
        """Test _construct_faq_text with general answer"""
        mock_init_ui.return_value = MagicMock()
        
        faq_ui = FAQSummaryFixUI(self.mock_config)
        
        # Mock the _generate_id method
        with patch.object(faq_ui, '_generate_id', return_value='test_id_456'):
            # Create test row
            row = pd.Series({
                'Q': 'How to reset password?',
                'A': 'You can reset your password by clicking the forgot password link.',
                'text': 'Q: How to reset password?\nA: You can reset your password by clicking the forgot password link.'
            })
            
            result_id, result_faq_text, result_rdb_text = faq_ui._construct_faq_text(
                row, 'Q', 'A', 'text',
                '<NO ANSWER/>', 
                'For more information about `{question}`, please visit the following link: [{question}](faq:{faq_id}).',
                '{output} For more details, please see the link [{question}](faq:{faq_id}).'
            )
            
            # Verify results
            self.assertEqual(result_id, 'test_id_456')
            expected_faq_text = 'Q: How to reset password?\nA: You can reset your password by clicking the forgot password link. For more details, please see the link [How to reset password?](faq:test_id_456).'
            self.assertEqual(result_faq_text, expected_faq_text)
            self.assertEqual(result_rdb_text, 'You can reset your password by clicking the forgot password link.')
    
    @patch('ui.faq_summary_fix_ui.FAQSummaryFixUI.init_ui')
    def test_construct_faq_text_with_q_prefix_in_text(self, mock_init_ui):
        """Test _construct_faq_text when text starts with 'Q: '"""
        mock_init_ui.return_value = MagicMock()
        
        faq_ui = FAQSummaryFixUI(self.mock_config)
        
        # Mock the _generate_id method
        with patch.object(faq_ui, '_generate_id', return_value='test_id_789'):
            # Create test row with text starting with 'Q: '
            row = pd.Series({
                'Q': 'What is your name?',
                'A': 'My name is Assistant.',
                'text': 'Q: What is your name?\nA: My name is Assistant.'
            })
            
            result_id, result_faq_text, result_rdb_text = faq_ui._construct_faq_text(
                row, 'Q', 'A', 'text',
                '<NO ANSWER/>', 
                'For more information about `{question}`, please visit the following link: [{question}](faq:{faq_id}).',
                '{output} For more details, please see the link [{question}](faq:{faq_id}).'
            )
            
            # Verify results
            self.assertEqual(result_id, 'test_id_789')
            expected_faq_text = 'Q: What is your name?\nA: My name is Assistant. For more details, please see the link [What is your name?](faq:test_id_789).'
            self.assertEqual(result_faq_text, expected_faq_text)
            # Should extract text after 'A: '
            self.assertEqual(result_rdb_text, 'My name is Assistant.')


class TestFAQSummaryFixUIDataProcessing(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
    
    @patch('ui.faq_summary_fix_ui.FAQSummaryFixUI.init_ui')
    def test_process_faq_data_success(self, mock_init_ui):
        """Test _process_faq_data successful processing using real implementation"""
        mock_init_ui.return_value = MagicMock()
        
        faq_ui = FAQSummaryFixUI(self.mock_config)
        
        # Create test dataframe
        test_df = pd.DataFrame({
            'Q': ['Question 1', 'Question 2'],
            'A': ['Answer 1', '<NO ANSWER/>'],
            'text': ['Q: Question 1\nA: Answer 1', 'Q: Question 2\nA: No answer available']
        })
        
        # Use real implementation but with mocked ID generation for predictable results
        with patch.object(faq_ui, '_generate_id', side_effect=['id_1', 'id_2']):
            result_df = faq_ui._process_faq_data(
                test_df, 'Q', 'A', 'text',
                'faq_id', 'faq_text', 'rdb_text',
                '<NO ANSWER/>', 
                'For more information about `{question}`, please visit the following link: [{question}](faq:{faq_id}).',
                '{output} For more details, please see the link [{question}](faq:{faq_id}).'
            )
            
            # Verify new columns were added
            self.assertIn('faq_id', result_df.columns)
            self.assertIn('faq_text', result_df.columns)
            self.assertIn('rdb_text', result_df.columns)
            
            # Verify we have the right number of rows
            self.assertEqual(len(result_df), 2)
            
            # Verify ID generation was called
            self.assertEqual(faq_ui._generate_id.call_count, 2)
    
    @patch('ui.faq_summary_fix_ui.FAQSummaryFixUI.init_ui')
    def test_process_faq_data_error(self, mock_init_ui):
        """Test _process_faq_data with processing error"""
        mock_init_ui.return_value = MagicMock()
        
        faq_ui = FAQSummaryFixUI(self.mock_config)
        
        # Create invalid dataframe (missing required columns)
        test_df = pd.DataFrame({
            'invalid_column': ['data1', 'data2']
        })
        
        with self.assertRaises(gr.Error):
            faq_ui._process_faq_data(
                test_df, 'Q', 'A', 'text',
                'faq_id', 'faq_text', 'rdb_text',
                '<NO ANSWER/>', 
                'No answer template',
                'General template'
            )


class TestFAQSummaryFixUIIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
    
    @patch('ui.faq_summary_fix_ui.dump_dataframe')
    @patch('ui.faq_summary_fix_ui.gr.Blocks')
    def test_full_integration_flow(self, mock_blocks, mock_dump_dataframe):
        """Test full integration of FAQSummaryFixUI workflow"""
        mock_ui = MagicMock()
        mock_blocks.return_value.__enter__.return_value = mock_ui
        mock_dump_dataframe.return_value = ['output_file.csv', 'output_file.xlsx']
        
        with patch('ui.faq_summary_fix_ui.gr.Row'), \
             patch('ui.faq_summary_fix_ui.gr.Column'), \
             patch('ui.faq_summary_fix_ui.gr.File'), \
             patch('ui.faq_summary_fix_ui.gr.Button'), \
             patch('ui.faq_summary_fix_ui.gr.Dataframe'), \
             patch('ui.faq_summary_fix_ui.gr.Accordion'), \
             patch('ui.faq_summary_fix_ui.gr.Textbox'):
            
            faq_ui = FAQSummaryFixUI(self.mock_config)
            
            # Verify all components exist
            self.assertIsNotNone(faq_ui.file_input)
            self.assertIsNotNone(faq_ui.download_output)
            self.assertIsNotNone(faq_ui.process_button)
            self.assertIsNotNone(faq_ui.data_before)
            self.assertIsNotNone(faq_ui.data_after)
            
            # Verify parameter fields exist
            self.assertIsNotNone(faq_ui.question_field)
            self.assertIsNotNone(faq_ui.summary_field)
            self.assertIsNotNone(faq_ui.q_and_a_field)
            self.assertIsNotNone(faq_ui.faq_id_output_field)
            self.assertIsNotNone(faq_ui.faq_text_output_field)
            self.assertIsNotNone(faq_ui.rdb_text_output_field)
            self.assertIsNotNone(faq_ui.no_answer_pattern)
            self.assertIsNotNone(faq_ui.no_answer_output_template)
            self.assertIsNotNone(faq_ui.general_output_template)
            
            # Verify UI object was returned
            self.assertEqual(faq_ui.ui, mock_ui)


if __name__ == '__main__':
    unittest.main()