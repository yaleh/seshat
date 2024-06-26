import argparse

import gradio as gr
from confz import FileSource, CLArgSource

from tools.config_loader import AppConfig
from ui import (
    BatchUI,
    MetaPromptUI,
    EmbeddingUI,
    LangServeClientUI,
    FAQSummaryFixUI,
    SystemUI
)

class ChatbotApp:
    def __init__(self, config=None, config_file_path=None):
        self.batch_ui = BatchUI(config)
        self.embedding_ui = EmbeddingUI(config)
        self.meta_ui = MetaPromptUI(config)
        self.langserve_client_ui = LangServeClientUI(config)
        self.faq_summary_fix_ui = FAQSummaryFixUI(config)
        self.system_ui = SystemUI(config, config_file_path)

        self.ui = gr.TabbedInterface(
            interface_list=[
                self.batch_ui.ui,
                self.langserve_client_ui.ui,
                self.embedding_ui.ui, 
                self.meta_ui.ui,
                self.faq_summary_fix_ui.ui,
                self.system_ui.ui
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='config.yaml',)
    parsed_args, _ = parser.parse_known_args()
    return parsed_args

if __name__ == "__main__":
    args = parse_args()

    config_sources = [
        FileSource(file=args.config_file),
        CLArgSource()
    ]
    app_config = AppConfig(config_sources=config_sources)

    app = ChatbotApp(app_config, args.config_file)
    app.ui.queue().launch(
        share=app_config.server.share,
        server_name=app_config.server.host,
        server_port=app_config.server.port,
        debug=True
    )