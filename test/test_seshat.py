import unittest
import argparse
from unittest.mock import patch, MagicMock, Mock
import sys

import seshat
from seshat import ChatbotApp, parse_args


class TestChatbotApp(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MagicMock()
        self.config_file_path = '/tmp/test_config.yaml'
    
    @patch('seshat.BatchUI')
    @patch('seshat.EmbeddingUI')
    @patch('seshat.MetaPromptUI')
    @patch('seshat.LangServeClientUI')
    @patch('seshat.FAQSummaryFixUI')
    @patch('seshat.SystemUI')
    @patch('seshat.gr.TabbedInterface')
    def test_chatbot_app_init(self, mock_tabbed_interface, mock_system_ui, 
                             mock_faq_ui, mock_langserve_ui, mock_meta_ui, 
                             mock_embedding_ui, mock_batch_ui):
        """Test ChatbotApp initialization"""
        # Setup mock UI instances
        mock_ui_instances = {}
        for ui_name, mock_ui in [
            ('batch_ui', mock_batch_ui),
            ('embedding_ui', mock_embedding_ui),
            ('meta_ui', mock_meta_ui),
            ('langserve_client_ui', mock_langserve_ui),
            ('faq_summary_fix_ui', mock_faq_ui),
            ('system_ui', mock_system_ui)
        ]:
            mock_instance = MagicMock()
            mock_instance.ui = f"{ui_name}_interface"
            mock_ui.return_value = mock_instance
            mock_ui_instances[ui_name] = mock_instance
        
        mock_tabbed_interface_instance = MagicMock()
        mock_tabbed_interface.return_value = mock_tabbed_interface_instance
        
        # Create ChatbotApp
        app = ChatbotApp(self.mock_config, self.config_file_path)
        
        # Verify all UI components were created with config
        mock_batch_ui.assert_called_once_with(self.mock_config)
        mock_embedding_ui.assert_called_once_with(self.mock_config)
        mock_meta_ui.assert_called_once_with(self.mock_config)
        mock_langserve_ui.assert_called_once_with(self.mock_config)
        mock_faq_ui.assert_called_once_with(self.mock_config)
        mock_system_ui.assert_called_once_with(self.mock_config, self.config_file_path)
        
        # Verify TabbedInterface was created with correct parameters
        mock_tabbed_interface.assert_called_once_with(
            interface_list=[
                'batch_ui_interface',
                'langserve_client_ui_interface',
                'embedding_ui_interface',
                'meta_ui_interface',
                'faq_summary_fix_ui_interface',
                'system_ui_interface'
            ],
            tab_names=[
                'Batch',
                'Lang Serve',
                'Embedding & VDB',
                'Meta Prompt',
                'FAQ Summary Fix',
                'System'
            ],
            title='Seshat'
        )
        
        # Verify app stores the UI components
        self.assertEqual(app.batch_ui, mock_ui_instances['batch_ui'])
        self.assertEqual(app.embedding_ui, mock_ui_instances['embedding_ui'])
        self.assertEqual(app.meta_ui, mock_ui_instances['meta_ui'])
        self.assertEqual(app.langserve_client_ui, mock_ui_instances['langserve_client_ui'])
        self.assertEqual(app.faq_summary_fix_ui, mock_ui_instances['faq_summary_fix_ui'])
        self.assertEqual(app.system_ui, mock_ui_instances['system_ui'])
        self.assertEqual(app.ui, mock_tabbed_interface_instance)
    
    @patch('seshat.BatchUI')
    @patch('seshat.EmbeddingUI')
    @patch('seshat.MetaPromptUI')
    @patch('seshat.LangServeClientUI')
    @patch('seshat.FAQSummaryFixUI')
    @patch('seshat.SystemUI')
    @patch('seshat.gr.TabbedInterface')
    def test_chatbot_app_init_none_config(self, mock_tabbed_interface, mock_system_ui,
                                         mock_faq_ui, mock_langserve_ui, mock_meta_ui,
                                         mock_embedding_ui, mock_batch_ui):
        """Test ChatbotApp initialization with None config"""
        # Setup mocks
        for mock_ui in [mock_batch_ui, mock_embedding_ui, mock_meta_ui, 
                       mock_langserve_ui, mock_faq_ui]:
            mock_ui.return_value.ui = MagicMock()
        mock_system_ui.return_value.ui = MagicMock()
        
        # Create ChatbotApp with None config
        app = ChatbotApp(None, None)
        
        # Verify all UI components were created with None config
        mock_batch_ui.assert_called_once_with(None)
        mock_embedding_ui.assert_called_once_with(None)
        mock_meta_ui.assert_called_once_with(None)
        mock_langserve_ui.assert_called_once_with(None)
        mock_faq_ui.assert_called_once_with(None)
        mock_system_ui.assert_called_once_with(None, None)


class TestParseArgs(unittest.TestCase):
    
    def test_parse_args_default(self):
        """Test parse_args with default arguments"""
        # Mock sys.argv to simulate no command line arguments
        with patch.object(sys, 'argv', ['seshat.py']):
            args = parse_args()
            
            self.assertEqual(args.config_file, 'config.yaml')
    
    def test_parse_args_custom_config_file(self):
        """Test parse_args with custom config file"""
        # Mock sys.argv to simulate custom config file argument
        test_config = '/path/to/custom_config.yaml'
        with patch.object(sys, 'argv', ['seshat.py', '--config_file', test_config]):
            args = parse_args()
            
            self.assertEqual(args.config_file, test_config)
    
    @patch('seshat.argparse.ArgumentParser.parse_known_args')
    def test_parse_args_with_unknown_args(self, mock_parse_known_args):
        """Test parse_args handles unknown arguments correctly"""
        # Mock parse_known_args to return args and unknown args
        mock_args = MagicMock()
        mock_args.config_file = 'test_config.yaml'
        mock_unknown = ['--unknown-arg', 'value']
        mock_parse_known_args.return_value = (mock_args, mock_unknown)
        
        result = parse_args()
        
        # Verify parse_known_args was called (handles unknown args)
        mock_parse_known_args.assert_called_once()
        self.assertEqual(result, mock_args)
    
    def test_parse_args_argument_parser_setup(self):
        """Test that ArgumentParser is set up correctly"""
        with patch('seshat.argparse.ArgumentParser') as mock_parser_class:
            mock_parser = MagicMock()
            mock_parser_class.return_value = mock_parser
            mock_parser.parse_known_args.return_value = (MagicMock(), [])
            
            parse_args()
            
            # Verify ArgumentParser was created
            mock_parser_class.assert_called_once()
            
            # Verify config_file argument was added
            mock_parser.add_argument.assert_called_once_with(
                "--config_file", 
                type=str, 
                default='config.yaml'
            )


class TestMainExecution(unittest.TestCase):
    
    def test_main_execution_components(self):
        """Test main execution components are properly defined"""
        # Test that main execution functions exist and work
        from seshat import parse_args, ChatbotApp
        from tools.config_loader import AppConfig
        from confz import FileSource, CLArgSource
        
        # Verify classes and functions exist
        self.assertIsNotNone(parse_args)
        self.assertIsNotNone(ChatbotApp)
        self.assertIsNotNone(AppConfig)
        self.assertIsNotNone(FileSource)
        self.assertIsNotNone(CLArgSource)
        
        # Test parse_args returns expected structure
        with patch('sys.argv', ['seshat.py']):
            args = parse_args()
            self.assertTrue(hasattr(args, 'config_file'))
    
    @patch('seshat.ChatbotApp')
    @patch('seshat.AppConfig')  
    def test_main_workflow_simulation(self, mock_app_config, mock_chatbot_app):
        """Test simulated main workflow"""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.server.share = True
        mock_config.server.host = '0.0.0.0'
        mock_config.server.port = 7860
        mock_app_config.return_value = mock_config
        
        mock_app = MagicMock()
        mock_ui = MagicMock()
        mock_app.ui = mock_ui
        mock_queue = MagicMock()
        mock_launch = MagicMock()
        mock_ui.queue.return_value.launch = mock_launch
        mock_chatbot_app.return_value = mock_app
        
        # Simulate main workflow
        from confz import FileSource, CLArgSource
        
        config_file = 'test_config.yaml'
        config_sources = [FileSource(file=config_file), CLArgSource()]
        app_config = mock_app_config(config_sources=config_sources)
        app = mock_chatbot_app(app_config, config_file)
        
        # Simulate launch call
        app.ui.queue().launch(
            share=app_config.server.share,
            server_name=app_config.server.host,
            server_port=app_config.server.port,
            debug=True
        )
        
        # Verify the workflow
        mock_app_config.assert_called()
        mock_chatbot_app.assert_called_with(app_config, config_file)


class TestImports(unittest.TestCase):
    
    def test_required_imports(self):
        """Test that all required modules can be imported"""
        # Test that seshat module imports successfully
        import seshat
        
        # Test that main classes are available
        self.assertTrue(hasattr(seshat, 'ChatbotApp'))
        self.assertTrue(hasattr(seshat, 'parse_args'))
        
        # Test that UI classes are imported
        from ui import (
            BatchUI, MetaPromptUI, EmbeddingUI, 
            LangServeClientUI, FAQSummaryFixUI, SystemUI
        )
        
        # Verify classes exist
        self.assertIsNotNone(BatchUI)
        self.assertIsNotNone(MetaPromptUI)
        self.assertIsNotNone(EmbeddingUI)
        self.assertIsNotNone(LangServeClientUI)
        self.assertIsNotNone(FAQSummaryFixUI)
        self.assertIsNotNone(SystemUI)


class TestChatbotAppIntegration(unittest.TestCase):
    
    @patch('seshat.gr.TabbedInterface')
    def test_chatbot_app_ui_structure(self, mock_tabbed_interface):
        """Test that ChatbotApp creates proper UI structure"""
        mock_config = MagicMock()
        
        with patch('seshat.BatchUI'), patch('seshat.EmbeddingUI'), \
             patch('seshat.MetaPromptUI'), patch('seshat.LangServeClientUI'), \
             patch('seshat.FAQSummaryFixUI'), patch('seshat.SystemUI'):
            
            app = ChatbotApp(mock_config, 'config.yaml')
            
            # Verify TabbedInterface call structure
            call_args = mock_tabbed_interface.call_args[1]
            
            # Check that we have the right number of interfaces and tab names
            self.assertEqual(len(call_args['interface_list']), 6)
            self.assertEqual(len(call_args['tab_names']), 6)
            self.assertEqual(call_args['title'], 'Seshat')
            
            # Verify tab names are in expected order
            expected_tabs = [
                'Batch', 'Lang Serve', 'Embedding & VDB', 
                'Meta Prompt', 'FAQ Summary Fix', 'System'
            ]
            self.assertEqual(call_args['tab_names'], expected_tabs)


if __name__ == '__main__':
    unittest.main()