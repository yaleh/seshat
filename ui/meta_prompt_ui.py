import gradio as gr
from langchain.schema import HumanMessage, SystemMessage
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# from prompts.prmmpt_prompts import *
# from components.chatbot import ChatbotFactory
from components.lcel import LLMModelFactory

class MetaPromptUI:
    def __init__(self, config):
        self.config = config

        self.chatbot_factory = LLMModelFactory()

        self.model_service_list = self.config.llm.default_model_service
        self.model_factory = LLMModelFactory()

        # generating llm
        self.generating_model_service = self.config.meta_prompt.default_meta_model_service
        self.generating_model_type = self.config.llm.llm_services[self.generating_model_service].type
        self.generating_model_args = self.config.llm.llm_services[self.generating_model_service].args.dict()
        self.generating_model_name = self.config.meta_prompt.default_meta_model_name
    
        self.generating_show_other_options = False

        # testing llm
        self.testing_model_service = self.config.meta_prompt.default_target_model_service
        self.testing_model_type = self.config.llm.llm_services[self.testing_model_service].type
        self.testing_model_args = self.config.llm.llm_services[self.testing_model_service].args.dict()
        self.testing_model_name = self.config.meta_prompt.default_target_model_name
    
        self.testing_show_other_options = False

        self.enable_other_user_prompts = False
        self.ui = self.init_ui()

    def init_ui(self):
        with gr.Blocks() as prompt_ui:
            with gr.Row():
                with gr.Column():
                    # Prompt 1: 根据input和output生成prompt
                    self.testing_user_prompt_textbox = gr.Textbox(
                            label="Testing User Prompt",
                            lines=10,
                            interactive=True,
                            show_copy_button=True
                            )
                    self.expect_output_textbox = gr.Textbox(
                            label="Expected Output",
                            lines=5,
                            interactive=True,
                            show_copy_button=True
                            )
                    self.other_user_prompts_checkbox = gr.Checkbox(
                            label="Other User Prompts",
                            info="Enable other user prompts in meta prompt?",
                            value=self.enable_other_user_prompts
                            )
                    self.other_user_prompts_textbox = gr.Textbox(
                        label="Other User Prompts",
                        lines=10,
                        interactive=True,
                        placeholder="Wrap each prompt with a pair of '```'.",
                        visible=self.enable_other_user_prompts,
                        show_copy_button=True
                        )
                    # Add gr.Number here for iterations input
                    self.iterations_number = gr.Number(value=1, label="Optimize Iterations")
                    # Add button to trigger optimization here
                    self.optimize_btn = gr.Button(value="Optimize Prompt", variant='primary')
                with gr.Column():
                    self.new_system_prompt_textbox = gr.Textbox(
                        label="New System Prompt",
                        lines=5,
                        interactive=True,
                        show_copy_button=True
                        )
                    self.new_system_prompt_changed = gr.Checkbox(
                        label="New System Prompt Changed",
                        value=False,
                        interactive=False
                        )
                    self.new_output_textbox = gr.Textbox(
                        label="New Output",
                        lines=5,
                        interactive=True,
                        show_copy_button=True
                        )
                    with gr.Row():
                        self.run_meta_btn = gr.Button(value="↑ Single Step Optimize")
                        self.run_new_btn = gr.Button(value="⟳ Run New")
                        self.run_new_btn.click(
                            fn=self.test_prompt,
                            inputs=[
                                self.new_system_prompt_textbox,
                                self.testing_user_prompt_textbox
                            ],
                            outputs=[self.new_output_textbox]
                        )
                with gr.Column():
                    self.current_system_prompt_textbox = gr.Textbox(
                        label="Current System Prompt",
                        value="",
                        lines=5,
                        interactive=True,
                        show_copy_button=True
                        )
                    self.current_output_textbox = gr.Textbox(
                        label="Current Output",
                        lines=5,
                        interactive=True,
                        show_copy_button=True
                        )
                    with gr.Row():
                        self.accept_new_btn = gr.Button(value="→ Accept New Prompt")
                        self.run_current_btn = gr.Button(value="⟳ Run Current")
                        self.accept_new_btn.click(
                            fn=self.copy_new_prompts,
                            inputs=[self.new_system_prompt_textbox, self.new_output_textbox],
                            outputs=[self.current_system_prompt_textbox, self.current_output_textbox]
                        )
                        self.run_current_btn.click(
                            fn=self.test_prompt,
                            inputs=[self.current_system_prompt_textbox, self.testing_user_prompt_textbox],
                            outputs=[self.current_output_textbox]
                        )

                    self.similar_candidate_textbox = gr.Textbox(
                        label="Similarity Delta Between New and Current Output",
                        lines=1,
                        interactive=True
                    )
                    self.compare_outputs_btn = gr.Button(value="Compare Outputs")
                    self.compare_outputs_btn.click(
                        self.compare_outputs,
                        [self.new_output_textbox, self.current_output_textbox, self.expect_output_textbox],
                        [self.similar_candidate_textbox]
                    )

            with gr.Row():
                with gr.Column():
                    # llm 服务
                    with gr.Row():
                        self.generating_llm_service_dropdown = gr.Dropdown(
                            label='Generating LLM Service',
                            choices=self.config.llm.llm_services.keys(),
                            value=self.generating_model_service,
                            interactive=True,
                            allow_custom_value=False
                        )
                        self.generating_llm_model_name_dropdown = gr.Dropdown(
                            label="Generating LLM Model Name",
                            choices=self.config.llm.llm_services[self.generating_model_service].models,
                            value=self.generating_model_name,
                            interactive=True,
                            allow_custom_value=False
                        )

                    self.generating_llm_service_dropdown.change(
                        self.update_llm_model_name_dropdown,
                        [self.generating_llm_service_dropdown],
                        [self.generating_llm_model_name_dropdown]
                    )
                    self.generating_llm_other_option_checkbox = gr.Checkbox(
                        value=False,
                        label="More Generating LLM Model Options"
                    )

                    self.generating_llm_model_temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.0,
                        interactive=True,
                        label="LLM Model Temperature",
                        visible=False
                    )
                    self.generating_llm_model_max_tokens_slider = gr.Slider(
                        minimum=0,
                        maximum=32000,
                        step=256,
                        value=0,
                        interactive=True,
                        label="LLM Model Token Limit (0 for auto)",
                        visible=False
                    )
                    self.generating_llm_model_request_timeout_slider = gr.Slider(
                        minimum=0,
                        maximum=600,
                        step=5,
                        value=600,
                        interactive=True,
                        label="LLM Model Timeout",
                        visible=False
                    )
                    self.generating_llm_model_max_retries_slider = gr.Slider(
                        minimum=0,
                        maximum=30,
                        step=1,
                        value=6,
                        interactive=True,
                        label="LLM Model Max Retries",
                        visible=False
                    )

                    self.generating_llm_other_option_checkbox.change(
                        self.show_other_llm_option,
                        inputs=[self.generating_llm_other_option_checkbox],
                        outputs=[
                            self.generating_llm_model_temperature_slider,
                            self.generating_llm_model_max_tokens_slider,
                            self.generating_llm_model_request_timeout_slider,
                            self.generating_llm_model_max_retries_slider
                        ]
                    )
                    self.generating_llm_model_name_dropdown.change(
                        fn=self.update_generating_llmbot,
                        inputs=[
                            self.generating_llm_service_dropdown,
                            self.generating_llm_model_name_dropdown,
                            self.generating_llm_model_temperature_slider,
                            self.generating_llm_model_max_tokens_slider,
                            self.generating_llm_model_request_timeout_slider,
                            self.generating_llm_model_max_retries_slider
                        ]
                    )
                    self.generating_llm_model_temperature_slider.change(
                        fn=self.update_generating_llmbot,
                        inputs=[
                            self.generating_llm_service_dropdown,
                            self.generating_llm_model_name_dropdown,
                            self.generating_llm_model_temperature_slider,
                            self.generating_llm_model_max_tokens_slider,
                            self.generating_llm_model_request_timeout_slider,
                            self.generating_llm_model_max_retries_slider
                        ]
                    )
                    self.generating_llm_model_max_tokens_slider.change(
                        fn=self.update_generating_llmbot,
                        inputs=[
                            self.generating_llm_service_dropdown,
                            self.generating_llm_model_name_dropdown,
                            self.generating_llm_model_temperature_slider,
                            self.generating_llm_model_max_tokens_slider,
                            self.generating_llm_model_request_timeout_slider,
                            self.generating_llm_model_max_retries_slider
                        ]
                    )
                    self.generating_llm_model_request_timeout_slider.change(
                        fn=self.update_generating_llmbot,
                        inputs=[
                            self.generating_llm_service_dropdown,
                            self.generating_llm_model_name_dropdown,
                            self.generating_llm_model_temperature_slider,
                            self.generating_llm_model_max_tokens_slider,
                            self.generating_llm_model_request_timeout_slider,
                            self.generating_llm_model_max_retries_slider
                        ]
                    )

                    # llm 服务
                    with gr.Row():
                        self.testing_llm_service_dropdown = gr.Dropdown(
                            label='Testing LLM Service',
                            choices=self.config.llm.llm_services.keys(),
                            value=self.testing_model_service,
                            interactive=True,
                            allow_custom_value=False
                        )
                        self.testing_llm_model_name_dropdown = gr.Dropdown(
                            label="Testing LLM Model Name",
                            choices=self.config.llm.llm_services[self.testing_model_service].models,
                            value=self.testing_model_name,
                            interactive=True,
                            allow_custom_value=False
                        )

                    self.testing_llm_service_dropdown.change(
                        self.update_llm_model_name_dropdown,
                        [self.testing_llm_service_dropdown],
                        [self.testing_llm_model_name_dropdown]
                    )
                    self.testing_llm_other_option_checkbox = gr.Checkbox(
                        value=False,
                        label="More Testing LLM Model Options"
                    )

                    self.testing_llm_model_temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.0,
                        interactive=True,
                        label="LLM Model Temperature",
                        visible=False
                    )
                    self.testing_llm_model_max_tokens_slider = gr.Slider(
                        minimum=0,
                        maximum=32000,
                        step=256,
                        value=0,
                        interactive=True,
                        label="LLM Model Token Limit (0 for auto)",
                        visible=False
                    )
                    self.testing_llm_model_request_timeout_slider = gr.Slider(
                        minimum=0,
                        maximum=600,
                        step=5,
                        value=600,
                        interactive=True,
                        label="LLM Model Timeout",
                        visible=False
                    )
                    self.testing_llm_model_max_retries_slider = gr.Slider(
                        minimum=0,
                        maximum=30,
                        step=1,
                        value=6,
                        interactive=True,
                        label="LLM Model Max Retries",
                        visible=False
                    )

                    self.testing_llm_other_option_checkbox.change(
                        self.show_other_llm_option,
                        inputs=[self.testing_llm_other_option_checkbox],
                        outputs=[
                            self.testing_llm_model_temperature_slider,
                            self.testing_llm_model_max_tokens_slider,
                            self.testing_llm_model_request_timeout_slider,
                            self.testing_llm_model_max_retries_slider
                        ]
                    )
                    self.testing_llm_model_name_dropdown.change(
                        fn=self.update_testing_llmbot,
                        inputs=[
                            self.testing_llm_service_dropdown,
                            self.testing_llm_model_name_dropdown,
                            self.testing_llm_model_temperature_slider,
                            self.testing_llm_model_max_tokens_slider,
                            self.testing_llm_model_request_timeout_slider,
                            self.testing_llm_model_max_retries_slider
                        ]
                    )
                    self.testing_llm_model_temperature_slider.change(
                        fn=self.update_testing_llmbot,
                        inputs=[
                            self.testing_llm_service_dropdown,
                            self.testing_llm_model_name_dropdown,
                            self.testing_llm_model_temperature_slider,
                            self.testing_llm_model_max_tokens_slider,
                            self.testing_llm_model_request_timeout_slider,
                            self.testing_llm_model_max_retries_slider
                        ]
                    )
                    self.testing_llm_model_max_tokens_slider.change(
                        fn=self.update_testing_llmbot,
                        inputs=[
                            self.testing_llm_service_dropdown,
                            self.testing_llm_model_name_dropdown,
                            self.testing_llm_model_temperature_slider,
                            self.testing_llm_model_max_tokens_slider,
                            self.testing_llm_model_request_timeout_slider,
                            self.testing_llm_model_max_retries_slider
                        ]
                    )
                    self.testing_llm_model_request_timeout_slider.change(
                        fn=self.update_testing_llmbot,
                        inputs=[
                            self.testing_llm_service_dropdown,
                            self.testing_llm_model_name_dropdown,
                            self.testing_llm_model_temperature_slider,
                            self.testing_llm_model_max_tokens_slider,
                            self.testing_llm_model_request_timeout_slider,
                            self.testing_llm_model_max_retries_slider
                        ]
                    )

                with gr.Column():
                    self.meta_system_prompt_textbox = gr.Textbox(label="Meta System Prompt",
                                                                value=self.config.meta_prompt.meta_system_prompt,
                                                                lines=10,
                                                                interactive=True,
                                                                show_copy_button=True
                                                                )
                with gr.Column():
                    self.merged_meta_prompt_textbox = gr.Textbox(label="Merged Meta System Prompt",
                                                                lines=10,
                                                                interactive=False,
                                                                show_copy_button=True
                                                                )
                    self.merge_prompt_btn = gr.Button(value="Merge Meta System Prompt")
                    self.merge_prompt_btn.click(fn=self.merge_meta_system_prompt,
                                                inputs=[
                                                    self.meta_system_prompt_textbox,
                                                    self.current_system_prompt_textbox,
                                                    self.other_user_prompts_textbox,
                                                    self.testing_user_prompt_textbox,
                                                    self.expect_output_textbox,
                                                    self.current_output_textbox,
                                                    self.other_user_prompts_checkbox],
                                                outputs=[self.merged_meta_prompt_textbox])
            self.other_user_prompts_checkbox.change(self.update_enable_other_user_prompts,
                                                    [self.other_user_prompts_checkbox],
                                                    [
                                                        self.other_user_prompts_textbox,
                                                        self.meta_system_prompt_textbox
                                                        ])
            self.optimize_btn.click(
                self.optimize_prompt,
                [
                    self.meta_system_prompt_textbox,
                    self.current_system_prompt_textbox,
                    self.testing_user_prompt_textbox,
                    self.other_user_prompts_textbox,
                    self.expect_output_textbox,
                    self.current_output_textbox,
                    self.iterations_number,
                    self.other_user_prompts_checkbox
                ],
                [self.new_system_prompt_textbox, self.new_system_prompt_changed])
            self.run_meta_btn.click(
                fn=self.meta_prompt,
                inputs=[self.meta_system_prompt_textbox,
                    self.current_system_prompt_textbox,
                    self.testing_user_prompt_textbox,
                    self.other_user_prompts_textbox,
                    self.expect_output_textbox,
                    self.current_output_textbox,
                    self.other_user_prompts_checkbox,],
                outputs=[self.new_system_prompt_textbox, self.new_system_prompt_changed])

        return prompt_ui

    def copy_new_prompts(self, system_prompt, output):
        """Copy prompts and output from new to current textboxes."""
        return system_prompt, output

    def update_generating_llmbot(self, llm_service,
                                    model_name,
                                    temperature,
                                    max_tokens,
                                    request_timeout,
                                    max_retries):
        
        model_service = self.config.llm.llm_services[llm_service]
        self.generating_model_type = model_service.type
        self.generating_model_args = model_service.args.dict()
        self.generating_model_name = model_name

        self.generating_model_args.update({
            'temperature': temperature,
            'request_timeout': request_timeout,
            'max_retries': max_retries
        })

        if temperature > 0:
            self.generating_model_args['temperature'] = temperature
        else:
            self.generating_model_args.pop('temperature')

    def update_testing_llmbot(self, llm_service,
                                    model_name,
                                    temperature,
                                    max_tokens,
                                    request_timeout,
                                    max_retries):
        model_service = self.config.llm.llm_services[llm_service]
        self.testing_model_type = model_service.type
        self.testing_model_args = model_service.args.dict()
        self.testing_model_name = model_name

        self.testing_model_args.update({
            'max_tokens': max_tokens,
            'request_timeout': request_timeout,
            'max_retries': max_retries
        })

        if temperature > 0:
            self.testing_model_args['temperature'] = temperature
        else:
            self.testing_model_args.pop('temperature')

    def test_prompt(self, system_prompt, user_prompt):
        try:
            # Create the prompt
            prompt = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            gr.Info(f"testing LLM Service: {self.testing_model_service}, name: {self.testing_model_name}")
            # Get the response from OpenAI
            llm = self.model_factory.create_model(self.testing_model_type, 
                                                self.testing_model_name, 
                                                **self.testing_model_args)
            gpt_response = llm.invoke(prompt)
            # Return the output to be placed in the output textbox
            return gpt_response.content
        except Exception as e:
            raise gr.Error(f"Error: {e}")

    def meta_prompt(self,
            meta_system_prompt,
            current_system_prompt,
            testing_user_prompt,
            other_user_prompts,
            expect_output,
            current_output,
            use_user_prompts):
        # Format the user message
        user_message = f'''
* Prompt Template
```
{current_system_prompt}
```
* User Message
```
{testing_user_prompt}
```
{'* Other User Messages' if use_user_prompts else ''}
{other_user_prompts if use_user_prompts else ''}
* Expected GPT Message
```
{expect_output}
```
* GPT Message
```
{current_output}
```
'''
        try:
            # Create the prompt
            prompt = [
                SystemMessage(content=meta_system_prompt),
                HumanMessage(content=user_message)
            ]
            gr.Info(f"generating LLM Service: {self.generating_model_service}, name: {self.generating_model_name}")
            # Get the response from OpenAI
            llm = self.model_factory.create_model(self.generating_model_type,
                                                self.generating_model_name,
                                                **self.generating_model_args)
            gpt_response = llm.invoke(prompt)

            updated_prompt = self.extract_updated_prompt(gpt_response.content)
            changed = not self.detect_no_change(gpt_response.content)

            # Return the output to be placed in the new system prompt textbox
            if updated_prompt:
                return updated_prompt, changed
            else:
                return gpt_response.content, changed
        except Exception as e:
            raise gr.Error(f"Error: {e}")

    def extract_updated_prompt(self, gpt_response):
        # Regular expression pattern to find the text enclosed
        pattern = "<!-- BEGIN OF PROMPT -->(.*?)<!-- END OF PROMPT -->"

        # Using search method to find the first occurrence of the pattern
        result = re.search(pattern, gpt_response, re.DOTALL)

        if result:
            s = result.group(1).strip("\n")
            if s.startswith("```") and s.endswith("```"):
                s = s[3:-3]
            return s # Return the matched string
        else:
            return None  # If no such pattern is found return None

    def detect_no_change(self, gpt_response):
        # Regular expression pattern to find the exact string
        pattern = "<!-- NO CHANGE TO PROMPT -->"

        # Using search method to find the occurrence of the pattern
        result = re.search(pattern, gpt_response)

        if result:
            return True  # If the pattern is found return True
        else:
            return False  # If no such pattern is found return False

    def optimize_prompt(self, meta_system_prompt,
                            current_system_prompt,
                            testing_user_prompt,
                            other_user_prompts,
                            expect_output,
                            current_output,
                            iterations,
                            user_other_user_prompts,
                            progress=gr.Progress()):
        changed = False
        # Iterate the specified number of times
        for i in progress.tqdm(range(int(iterations)), desc="Iteration Process"):
        # for i in range(int(iterations)):
            # If current_output is None or not provided, get it from test_prompt
            if current_output is None or current_output=='':
                current_output = self.test_prompt(current_system_prompt, testing_user_prompt)
            new_prompt, changed = self.meta_prompt(
                meta_system_prompt,
                current_system_prompt,
                testing_user_prompt,
                other_user_prompts,
                expect_output,
                current_output,
                user_other_user_prompts)

            # If changed is False, break the loop
            if not changed:
                break

            # If there is an updated prompt and it's different from the current one, update current_system_prompt
            if new_prompt and new_prompt != current_system_prompt:
                current_system_prompt = new_prompt
                # Reset current_output to None so it gets recalculated in the next iteration
                current_output = None

        # current_system_prompt = current_system_prompt
        return current_system_prompt, changed  # Return the optimized system prompt

    def show_other_llm_option(self, visible_flag):
        return (
            gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.0,
                interactive=True,
                label="LLM Model Temperature",
                visible=visible_flag
            ),
            gr.Slider(
                minimum=0,
                maximum=32000,
                step=256,
                value=0,
                interactive=True,
                label="LLM Model Token Limit (0 for auto)",
                visible=visible_flag
            ),
            gr.Slider(
                minimum=0,
                maximum=600,
                step=5,
                value=600,
                interactive=True,
                label="LLM Model Timeout",
                visible=visible_flag
            ),
            gr.Slider(
                minimum=0,
                maximum=30,
                step=1,
                value=6,
                interactive=True,
                label="LLM Model Max Retries",
                visible=visible_flag
            )
        )

    def update_llm_model_name_dropdown(self, cur_llm_service):
        # self.model_name_list = self.config_loader.get_model_name_list(cur_llm_service)
        # return gr.Dropdown(choices=self.model_name_list)
        return gr.Dropdown(choices=self.config.llm.llm_services[cur_llm_service].models)

    def update_enable_other_user_prompts(self, new_value):
        self.enable_other_user_prompts = new_value
        return (
            gr.Textbox(
                label="Other User Prompts",
                lines=10,
                interactive=True,
                placeholder="Wrap each prompt with a pair of '```'.",
                visible=new_value,
                show_copy_button=True
            ),
            gr.Textbox(label="Meta System Prompt",
                       value=self.config.meta_prompt.meta_system_prompt_with_other_prompts
                       if new_value
                       else self.config.meta_prompt.meta_system_prompt,
                       lines=10,
                       interactive=True,
                       show_copy_button=True
                       )
        )

    def compare_strings(self, alpha: str, beta: str, expected: str) -> str:
        # If both ALPHA and BETA are empty, return None
        if not alpha and not beta:
            return None

        # If either ALPHA or BETA is empty, the non-empty string should be considered more similar to EXPECTED
        if not alpha:
            return 'B'
        if not beta:
            return 'A'

        # If both ALPHA and BETA are identical, return None
        if alpha == beta:
            return None

        # Create the CountVectorizer instance
        vectorizer = CountVectorizer().fit_transform([alpha, beta, expected])
        vectors = vectorizer.toarray()

        # Calculate cosine similarities
        alpha_sim = cosine_similarity(vectors[0].reshape(1, -1), vectors[2].reshape(1, -1))
        beta_sim = cosine_similarity(vectors[1].reshape(1, -1), vectors[2].reshape(1, -1))

        # Compare similarities and return the string that is more similar to the expected string
        if alpha_sim > beta_sim:
            return 'A'
        elif beta_sim > alpha_sim:
            return 'B'
        else:
            return None

    def delta_similarities(self, alpha: str, beta: str, expected: str) -> float:
        # If both ALPHA and BETA are empty, return 0
        if not alpha and not beta:
            return 0.0

        # If either ALPHA or BETA is empty, the non-empty string should be considered more similar to EXPECTED
        if not alpha:
            return -1.0
        if not beta:
            return 1.0

        # If both ALPHA and BETA are identical, return 0
        if alpha == beta:
            return 0.0

        # Create the CountVectorizer instance
        vectorizer = CountVectorizer().fit_transform([alpha, beta, expected])
        vectors = vectorizer.toarray()

        # Calculate cosine similarities
        alpha_sim = cosine_similarity(vectors[0].reshape(1, -1), vectors[2].reshape(1, -1))
        beta_sim = cosine_similarity(vectors[1].reshape(1, -1), vectors[2].reshape(1, -1))

        # Return the difference in similarities
        return alpha_sim[0][0] - beta_sim[0][0]

    def compare_outputs(self, new_output, current_output, expected_output):
        # Compare new output and current output against expected output
        # result = self.compare_strings(new_output, current_output, expected_output)
        result = self.delta_similarities(new_output, current_output, expected_output)
        return result

    def merge_meta_system_prompt(
            self,
            meta_system_prompt,
            current_system_prompt,
            other_user_prompts,
            testing_user_prompt,
            expect_output,
            current_output,
            use_other_user_prompts
            ):
        """Merge meta and current system prompts."""
        # converted_prompts = [prompt[0] for prompt in other_user_prompts.values]
        user_prompt = f'''
* Prompt Template
```
{current_system_prompt}
```
* User Message
```
{testing_user_prompt}
```
{'* Other User Messages' if use_other_user_prompts else ''}
{other_user_prompts if use_other_user_prompts else ''}
* Expected GPT Message
```
{expect_output}
```
* GPT Message
```
{current_output}
```
'''

        merged_prompt = f"{meta_system_prompt}{user_prompt}"

        return merged_prompt
