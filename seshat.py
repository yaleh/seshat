import argparse

import gradio as gr
from confz import FileSource, CLArgSource

from tools.config_loader import AppConfig
from ui import BatchUI, MetaPromptUI

class ChatbotApp:
    def __init__(self, config=None):
        self.batch_ui = BatchUI(config)
        self.meta_ui = MetaPromptUI(config)

        self.ui = gr.TabbedInterface(
            interface_list=[self.batch_ui.ui, self.meta_ui.ui],
            tab_names=['Batch', 'Meta Prompt'],
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

    app = ChatbotApp(app_config)
    app.ui.queue().launch(
        share=app_config.server.share,
        server_name=app_config.server.host,
        server_port=app_config.server.port,
        debug=True
    )