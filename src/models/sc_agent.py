
from config.configurator import configs
from models.base_model import BaseModel
import re
import json
import importlib
from data_utils.data_handler_re import DataHandlerRE
from modules.module_utils import format_sample_str
from modules.prompt.prompter import PrompterSCAgent

SUCCESS = 0
NO_RESULT_WITHIN_MAX_ITERATIONS = -1
NO_VALID_RESULT_WITHIN_MAX_RETRY = -2


class SC_Agent(BaseModel):
    mode: str = configs["model"]["mode"] if "mode" in configs["model"] else "dummy"
    stop: str = ["Output:", "Observation:"]        # LLM stop
    max_iterations: int = configs["model"]["max_iterations"]
    max_retry: int = configs["model"]["max_retry"]
    history: list = []
    tools: dict = {}
    prompter: PrompterSCAgent

    def __init__(self, data_handler):
        super().__init__(data_handler)
        if configs['train']['if_predict'] or configs['train']['if_train']:
            self.init_tools()
        self.prompter = PrompterSCAgent(data_handler)

    def init_tools(self):
        tools_activated = []
        for tool_name in configs['tools'].keys():
            if configs['tools'][tool_name]['open']:
                tools_activated.append(tool_name)
        self.logger.info(f"Activated tools: {tools_activated}")
        module = importlib.import_module('modules.tools')
        for tool_name in tools_activated:
            tool = getattr(module, tool_name)(self.data_handler)
            self.tools[tool_name] = tool
        self.logger.info(f"Tools: {self.tools}")

    def extract(self, text, idx):
        if configs['model']['mode'] == 'direct':
            return self.extract_direct(text, idx)
        elif configs['model']['mode'] == 'staged':
            return self.extract_staged(text, idx)
        else:
            raise ValueError(f"Unknown model mode: {configs['model']['model']}")

    # @log_exceptions
    def extract_staged(self, text, idx):
        debug = True

        text = json.dumps(text.strip(), ensure_ascii=False)
        if debug:
            self.logger.info(f"[idx={idx}] Input: {text}")
        self.history = []

        # TODO: Change iterations to 2, because we only allow this iteration if the model wants to make a tool call
        for _ in range(self.max_iterations): 
            system_prompt, user_prompt = self.generate_scagent_prompts(text)
            if self.history:
                user_prompt = self.extend_scagent_prompt(user_prompt)
            if idx < 5:
                self.log_prompt(system_prompt + "\n" + user_prompt)
            for _ in range(self.max_retry): # Do we even want this? 
                llm_output = self.query_llm_scagent(
                    system_prompt, user_prompt, stop=self.stop, temperature=0)
                if configs['data']['input_trace']:
                    self.llm_inputs.append({
                        "idx": idx,
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "llm_output": llm_output,
                    })
                err_code, parsed_res = self.parse_output(llm_output)
                if err_code == -1:
                    if debug:
                        self.logger.error(
                            f"error in parse_output: {llm_output}")
                    continue
                thought, action_name, args = parsed_res
                if action_name not in self.tools:
                    if debug:
                        self.logger.error(
                            f"error action_name: {action_name}. llm_output: {llm_output}")
                    continue
                if action_name == "Finish":
                    err_code, spo_list = self.parse_llm_output(args)
                    if err_code == -1:
                        if debug:
                            self.logger.error(
                                f"error in parse_llm_output: {args}. llm_output: {llm_output}")
                        continue
                break
            else:
                self.logger.error(
                    f"[ERROR] [Inner Loop] Failed to generate valid output after {self.max_retry} iterations.")
                return {
                    "spo_list_pred": [],
                    "history": self.history.copy(),
                    "final_output": llm_output,
                    "errorCode": NO_VALID_RESULT_WITHIN_MAX_RETRY,
                }

            self.history.append(f"Thought: {thought}")
            if debug:
                self.logger.info(f"Thought: {thought}")
            if action_name == "Finish":
                err_code, spo_list = self.parse_llm_output(args)

                finish_output = json.dumps(args, ensure_ascii=False)
                self.history.append(f"Finish: {finish_output}")
                if debug:
                    self.logger.info(f"Finish: {finish_output}")
                return {
                    "spo_list_pred": spo_list,
                    "history": self.history.copy(),
                    "final_output": llm_output,
                    "errorCode": err_code,
                }
            else:
                observation = self.tools[action_name].call(args)
                self.history.append(f"Action: {action_name}")
                self.history.append(f"ActionInput: {args}")
                if action_name == 'RetrieveExamples' and observation:
                    observation = "\n\n" + self.prompter.format_retrieved_examples(json.loads(observation))
                self.history.append(f"Observation: {observation}")
                if debug:
                    self.logger.info(f"Action: {action_name}")
                    self.logger.info(f"ActionInput: {args}")
                    self.logger.info(f"Observation: {observation}")
        else:
            self.logger.error(
                f"[ERROR] [Outer Loop] Failed to generate valid output after {self.max_iterations} iterations.")
            return {
                "spo_list_pred": [],
                "history": self.history.copy(),
                "final_output": llm_output,
                "errorCode": NO_RESULT_WITHIN_MAX_ITERATIONS,
            }

    def generate_scagent_prompts(self, text): 
        if configs['model']['mode'] == 'direct':
            system_prompt = self.prompter.get_system_prompt_c()
        else:
            system_prompt = self.prompter.get_system_prompt()
        if configs['model']['retrieval_switch']:
            retrieved_examples = json.loads(self.tools['RetrieveExamples'].call(text))
        else: 
            retrieved_examples = json.loads(self.tools['RetrieveExamples'].call_static_example())
        if retrieved_examples:
            # Use only 1 example for the base prompt and let the model decide if it needs more
            retrieved_examples = self.prompter.format_retrieved_examples([retrieved_examples[0]])
            # retrieved_examples = self.prompter.format_retrieved_examples(retrieved_examples)
        if configs['model']['retrieval_switch']:
            user_prompt = self.prompter.get_user_prompt(retrieved_examples, text)
        else:
            if configs['model']['no_examples']:
                user_prompt = self.prompter.get_static_user_prompt_no_examples(text)
            else: 
                user_prompt = self.prompter.get_static_user_prompt(retrieved_examples, text)
        return system_prompt, user_prompt
    
    def extend_scagent_prompt(self, user_prompt):
        for history in self.history:
            user_prompt += history + "\n"
        return user_prompt


    def parse_output(self, llm_output: str):
        try:
            # regex = r"(.*?)\nAction:(.*?)\nActionInput:[\s]*(.*)"
            regex = r"(.*)Action:(.*?)\nActionInput:[\s]*(.*)"
            match = re.search(regex, llm_output, re.DOTALL)
            thought = match.group(1).strip()
            action = match.group(2).strip()
            args = match.group(3).strip()
            thought = json.dumps(thought, ensure_ascii=False)
            return 0, (thought, action, args)

        except Exception as e:
            return -1, None

    def extract_direct(self, text, idx):
        debug = True

        text = json.dumps(text.strip(), ensure_ascii=False)
        if debug:
            self.logger.info(f"[idx={idx}] Input: {text}")

        system_prompt, user_prompt = self.generate_scagent_prompts(text)
        if idx < 5:
            self.log_prompt(system_prompt + "\n" + user_prompt)

        for _ in range(self.max_retry):
            llm_output = self.query_llm_scagent(
                system_prompt, user_prompt, stop=self.stop, temperature=0)
            
            if configs['data']['input_trace']:
                self.llm_inputs.append({
                    "idx": idx,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "llm_output": llm_output,
                })
            
            err_code, spo_list = self.parse_llm_output(llm_output)
            if err_code == -1:
                if debug:
                    self.logger.error(f"error in parse_llm_output: {llm_output}")
                continue
                
            if debug:
                self.logger.info(f"Final output: {llm_output}")
            return {
                "spo_list_pred": spo_list,
                "history": [llm_output],
                "final_output": llm_output,
                "errorCode": err_code,
            }
            
        self.logger.error(
            f"[ERROR] [Direct Extraction] Failed to generate valid output after {self.max_retry} retries.")
        return {
            "spo_list_pred": [],
            "history": [],
            "final_output": llm_output if 'llm_output' in locals() else "",
            "errorCode": NO_VALID_RESULT_WITHIN_MAX_RETRY,
        }
