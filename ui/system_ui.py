import gradio as gr
import os
import signal
import requests

class SystemUI:
    def __init__(self, config=None):
        self.config = config
        self.ui = self.init_ui()

    def init_ui(self):
        with gr.Blocks() as ui:
            # Todo:
            # * Update Cloudflare models

            # a button to exit the process
            self.refresh_openrouter_button = gr.Button("Refresh OpenRouter", variant="secondary")
            self.exit_button = gr.Button("Exit", variant="stop")
        
            self.refresh_openrouter_button.click(
                self.refresh_openrouter_services, [], []
            )

            self.exit_button.click(
                self.stop_server,[],[]
            )

        return ui
    
    def stop_server(self):
        os.kill(os.getpid(), signal.SIGTERM)
    
    def refresh_openrouter_services(self):
        # update self.config.llm.llm_services['OpenRouter'].models from https://openrouter.ai/api/v1/models

        # read json from https://openrouter.ai/api/v1/models
        response = requests.get("https://openrouter.ai/api/v1/models", timeout=10)  # Add timeout argument
        data = response.json()

        # example:
        # {"data":[{"id":"openrouter/auto","name":"Auto (best for prompt)","description":"Depending on their size, subject, and complexity, your prompts will be sent to [Mistral Large](/models/mistralai/mistral-large), [Claude 3 Sonnet](/models/anthropic/claude-3-sonnet:beta) or [GPT-4o](/models/openai/gpt-4o).  To see which model was used, visit [Activity](/activity).","pricing":{"prompt":"-1","completion":"-1","request":"-1","image":"-1"},"context_length":200000,"architecture":{"modality":"text","tokenizer":"Router","instruct_type":null},"top_provider":{"max_completion_tokens":null,"is_moderated":false},"per_request_limits":null},{"id":"nousresearch/nous-capybara-7b:free","name":"Nous: Capybara 7B (free)","description":"The Capybara series is a collection of datasets and models made by fine-tuning on data created by Nous, mostly in-house.\n\nV1.9 uses unalignment techniques for more consistent and dynamic control. It also leverages a significantly better foundation model, [Mistral 7B](/models/mistralai/mistral-7b-instruct).\n\nNote: this is a free, rate-limited version of [this model](/models/nousresearch/nous-capybara-7b). Outputs may be cached. Read about rate limits [here](/docs#limits).","pricing":{"prompt":"0","completion":"0","image":"0","request":"0"},"context_length":4096,"architecture":{"modality":"text","tokenizer":"Mistral","instruct_type":"airoboros"},"top_provider":{"max_completion_tokens":null,"is_moderated":false},"per_request_limits":{"prompt_tokens":"Infinity","completion_tokens":"Infinity"}}

        # process the data, get ids and set them to self.config.llm.llm_services['OpenRouter'].models
        openrouter_models = {}
        for model in data['data']:
            openrouter_models[model['id']] = model['name']
        self.config.llm.llm_services['OpenRouter'].models = openrouter_models

        gr.Info("OpenRouter services refreshed")        