import argparse

import gradio as gr
from confz import FileSource

from tools.config_loader import AppConfig
from ui import BatchUI

class ChatbotApp:
    def __init__(self, config=None):
        self.batch_ui = BatchUI(config)

        self.ui = gr.TabbedInterface(
            interface_list=[self.batch_ui.ui],
            tab_names=['Batch'],
            title='Seshat'
        )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--message_db", type=str, default="messages.db",
                        help="Filename for the DatabaseManager")
    parser.add_argument("--server_name", type=str, default="127.0.0.1", help="Server name or IP address")
    parser.add_argument("--share", action='store_true',
                        help="Launch app with sharing option")
    return parser.parse_args()

CONFIG_FILE = 'configs/custom_llm_config_v2.yaml'

if __name__ == "__main__":
    args = parse_args()

    config_sources = [
        FileSource(file=CONFIG_FILE, optional=True),
    ]
    app_config = AppConfig(config_sources=config_sources)

    app = ChatbotApp(app_config)
    app.ui.queue().launch(share=args.share, server_name=args.server_name, debug=True)