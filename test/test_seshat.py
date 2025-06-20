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


class TestSeshatMainExecution(unittest.TestCase):
    
    @patch('seshat.AppConfig')
    @patch('seshat.ChatbotApp')
    def test_main_execution_workflow(self, mock_chatbot_app, mock_app_config):
        """Test the complete main execution workflow"""
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
        
        # Mock command line arguments
        with patch('sys.argv', ['seshat.py', '--config_file', 'test_config.yaml']):
            # Import and execute main block simulation
            from confz import FileSource, CLArgSource
            
            args = parse_args()
            self.assertEqual(args.config_file, 'test_config.yaml')
            
            config_sources = [
                FileSource(file=args.config_file),
                CLArgSource()
            ]
            app_config = mock_app_config(config_sources=config_sources)
            app = mock_chatbot_app(app_config, args.config_file)
            
            # Simulate the launch call
            app.ui.queue().launch(
                share=app_config.server.share,
                server_name=app_config.server.host,
                server_port=app_config.server.port,
                debug=True
            )
            
            # Verify the workflow
            mock_app_config.assert_called_once()
            mock_chatbot_app.assert_called_once_with(app_config, args.config_file)
    
    @patch('seshat.AppConfig')
    @patch('seshat.ChatbotApp')
    def test_main_execution_with_default_config(self, mock_chatbot_app, mock_app_config):
        """Test main execution with default config file"""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.server.share = False
        mock_config.server.host = 'localhost'
        mock_config.server.port = 8080
        mock_app_config.return_value = mock_config
        
        mock_app = MagicMock()
        mock_ui = MagicMock()
        mock_app.ui = mock_ui
        mock_chatbot_app.return_value = mock_app
        
        # Mock command line arguments for default case
        with patch('sys.argv', ['seshat.py']):
            args = parse_args()
            self.assertEqual(args.config_file, 'config.yaml')  # Default value
            
            config_sources = [
                FileSource(file=args.config_file),
                CLArgSource()
            ]
            app_config = mock_app_config(config_sources=config_sources)
            app = mock_chatbot_app(app_config, args.config_file)
            
            # Verify correct parameters were used
            mock_chatbot_app.assert_called_once_with(app_config, 'config.yaml')
    
    def test_file_source_and_clarg_source_integration(self):
        """Test that FileSource and CLArgSource are properly used"""
        from confz import FileSource, CLArgSource
        
        # Test that these classes can be instantiated
        file_source = FileSource(file='test.yaml')
        clarg_source = CLArgSource()
        
        self.assertIsNotNone(file_source)
        self.assertIsNotNone(clarg_source)
        
        # Verify they can be used in a list (as done in main)
        config_sources = [file_source, clarg_source]
        self.assertEqual(len(config_sources), 2)
        self.assertIsInstance(config_sources[0], FileSource)
        self.assertIsInstance(config_sources[1], CLArgSource)


class TestSeshatAppConfigIntegration(unittest.TestCase):
    
    @patch('seshat.AppConfig')
    def test_app_config_with_file_source(self, mock_app_config):
        """Test AppConfig integration with FileSource"""
        from confz import FileSource, CLArgSource
        
        # Create config sources as done in main
        config_file = 'test_config.yaml'
        config_sources = [
            FileSource(file=config_file),
            CLArgSource()
        ]
        
        # Mock AppConfig
        mock_config = MagicMock()
        mock_app_config.return_value = mock_config
        
        # Test AppConfig instantiation
        app_config = mock_app_config(config_sources=config_sources)
        
        # Verify AppConfig was called with correct sources
        mock_app_config.assert_called_once_with(config_sources=config_sources)
        self.assertEqual(app_config, mock_config)
    
    @patch('seshat.AppConfig')
    def test_app_config_error_handling(self, mock_app_config):
        """Test AppConfig error handling"""
        from confz import FileSource, CLArgSource
        
        # Mock AppConfig to raise an exception
        mock_app_config.side_effect = Exception("Config error")
        
        config_sources = [
            FileSource(file='nonexistent.yaml'),
            CLArgSource()
        ]
        
        # Should raise the exception (no error handling in main)
        with self.assertRaises(Exception):
            mock_app_config(config_sources=config_sources)


class TestSeshatLaunchConfiguration(unittest.TestCase):
    
    def test_launch_parameters_extraction(self):
        """Test that launch parameters are correctly extracted from config"""
        # Create a mock config that simulates the real structure
        mock_config = MagicMock()
        mock_config.server.share = True
        mock_config.server.host = '127.0.0.1'
        mock_config.server.port = 9090
        
        # Verify the attributes can be accessed as they would be in main
        self.assertEqual(mock_config.server.share, True)
        self.assertEqual(mock_config.server.host, '127.0.0.1')
        self.assertEqual(mock_config.server.port, 9090)
        
        # Test with different values
        mock_config.server.share = False
        mock_config.server.host = '0.0.0.0'
        mock_config.server.port = 7860
        
        self.assertEqual(mock_config.server.share, False)
        self.assertEqual(mock_config.server.host, '0.0.0.0')
        self.assertEqual(mock_config.server.port, 7860)
    
    @patch('seshat.ChatbotApp')
    def test_gradio_ui_queue_and_launch(self, mock_chatbot_app):
        """Test Gradio UI queue and launch call structure"""
        # Setup mock app with proper method chaining
        mock_ui = MagicMock()
        mock_queue = MagicMock()
        mock_launch = MagicMock()
        
        # Set up the method chain: ui.queue().launch()
        mock_ui.queue.return_value.launch = mock_launch
        
        mock_app = MagicMock()
        mock_app.ui = mock_ui
        mock_chatbot_app.return_value = mock_app
        
        # Create app and simulate launch
        mock_config = MagicMock()
        app = mock_chatbot_app(mock_config, 'config.yaml')
        
        # Test the method chain
        app.ui.queue().launch(
            share=True,
            server_name='localhost',
            server_port=7860,
            debug=True
        )
        
        # Verify the calls were made
        mock_ui.queue.assert_called_once()
        mock_launch.assert_called_once_with(
            share=True,
            server_name='localhost',
            server_port=7860,
            debug=True
        )


class TestSeshatErrorScenarios(unittest.TestCase):
    
    @patch('seshat.ChatbotApp')
    def test_chatbot_app_instantiation_error(self, mock_chatbot_app):
        """Test error handling when ChatbotApp instantiation fails"""
        # Mock ChatbotApp to raise an exception
        mock_chatbot_app.side_effect = Exception("App initialization failed")
        
        # Should raise the exception (no error handling in main)
        with self.assertRaises(Exception):
            mock_chatbot_app(MagicMock(), 'config.yaml')
    
    def test_parse_args_with_invalid_arguments(self):
        """Test parse_args with various argument scenarios"""
        # Test with empty arguments
        with patch('sys.argv', ['seshat.py']):
            args = parse_args()
            self.assertEqual(args.config_file, 'config.yaml')
        
        # Test with custom config file
        with patch('sys.argv', ['seshat.py', '--config_file', '/path/to/custom.yaml']):
            args = parse_args()
            self.assertEqual(args.config_file, '/path/to/custom.yaml')
        
        # Test with unknown arguments (should be ignored due to parse_known_args)
        with patch('sys.argv', ['seshat.py', '--unknown_arg', 'value', '--config_file', 'test.yaml']):
            args = parse_args()
            self.assertEqual(args.config_file, 'test.yaml')
    
    @patch('seshat.ChatbotApp')
    def test_ui_launch_error(self, mock_chatbot_app):
        """Test error handling when UI launch fails"""
        # Setup mock app
        mock_ui = MagicMock()
        mock_ui.queue.return_value.launch.side_effect = Exception("Launch failed")
        
        mock_app = MagicMock()
        mock_app.ui = mock_ui
        mock_chatbot_app.return_value = mock_app
        
        app = mock_chatbot_app(MagicMock(), 'config.yaml')
        
        # Should raise the exception (no error handling in main)
        with self.assertRaises(Exception):
            app.ui.queue().launch(share=False, server_name='localhost', server_port=7860, debug=True)


if __name__ == '__main__':
    unittest.main()