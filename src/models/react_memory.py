
from config.configurator import configs
from models.base_model import BaseModel
import re
import json
import importlib
from modules.tools import MEMORY_NAME2TOOL
from trainer.metrics_v2 import EvaluatorRE
from modules.memory.memory import CorrectMemory, BaseMemory, ReflexionMemory, IncorrectMemory
from data_utils.data_handler_re import DataHandlerRE
from modules.module_utils import format_sample_str, format_reflexion_sample_str, format_incorrect_sample_str
from modules.prompt.prompter import PrompterReActMemory

SUCCESS = 0
NO_RESULT_WITHIN_MAX_ITERATIONS = -1
NO_VALID_RESULT_WITHIN_MAX_RETRY = -2


class ReAct_Memory(BaseModel):
    mode: str = configs["model"]["mode"] if "mode" in configs["model"] else "dummy"
    stop: str = ["Output:", "Observation:"]        # LLM stop
    max_iterations: int = configs["model"]["max_iterations"]
    max_retry: int = configs["model"]["max_retry"]
    num_pre_history: int = configs["model"]["num_pre_history"]
    use_summary: bool = configs["model"]["use_summary"]
    debug: bool = configs["model"]["debug"]

    history: list = []              # for recording the history, CLEARED in each iteration
    tools: dict = {}                # list of tools
    memory_names: list = []         # list of memories
    prompter: PrompterReActMemory   #

    # is_training = configs['train']['if_train']
    evaluator: EvaluatorRE = EvaluatorRE()

    def __init__(self, data_handler: DataHandlerRE):
        super().__init__(data_handler)
        if configs['train']['if_predict'] or configs['train']['if_train']:
            self.init_memorys()
            self.init_tools()
        self.prompter = PrompterReActMemory(data_handler)

    def init_tools(self):
        tools_activated = []
        for tool_name in configs['tools'].keys():
            if configs['tools'][tool_name]['open']:
                # NEW: remove RetrieveExamples
                if tool_name in ["RetrieveExamples"]:
                    continue
                tools_activated.append(tool_name)
        # NEW: set tools according to memory
        for memory_name in self.memory_names:
            tools_activated.append(MEMORY_NAME2TOOL[memory_name])
        self.logger.info(f"Activated tools: {tools_activated}")

        module = importlib.import_module('modules.tools')
        for tool_name in tools_activated:
            tool = getattr(module, tool_name)(self.data_handler)
            self.tools[tool_name] = tool
        self.logger.info(f"Tools: {self.tools}")

    # NEW: init memory
    def init_memorys(self):
        if configs['memory']['CorrectMemory']['open']:
            self.memory_names.append('CorrectMemory')
            self.data_handler.correct_memory = CorrectMemory()

            # init correct memory with few-shot
            num_samples_init = configs['memory']['CorrectMemory']['num_samples_init']
            if num_samples_init > 0:
                self.logger.info(
                    f"Init correct memory with {num_samples_init} samples.")
                samples_ds = self.data_handler.ds_index.select(
                    range(num_samples_init))
                samples_list = [samples_ds[i] for i in range(num_samples_init)]
                self.record_correct_memory(samples_list)

        if configs['memory']['IncorrectMemory']['open']:
            self.memory_names.append('IncorrectMemory')
            self.data_handler.incorrect_memory = IncorrectMemory()

        if configs['memory']['ReflexionMemory']['open']:
            self.memory_names.append('ReflexionMemory')
            self.data_handler.reflexion_memory = ReflexionMemory()

    # @log_exceptions
    def extract(self, text, idx):
        debug = True

        text = json.dumps(text.strip(), ensure_ascii=False)
        if debug:
            self.logger.info(f"[idx={idx}] Input: {text}")
        history = []       # clear the history!

        # ReAct-fashion
        for _ in range(self.max_iterations):
            prompt = self.generate_prompt(text, inference=True)
            if idx < 5:
                self.log_prompt(prompt)
            for _ in range(self.max_retry):
                # can try different parameters
                llm_output = self.query_llm(
                    prompt, stop=self.stop, temperature=0)
                if configs['data']['input_trace']:
                    self.llm_inputs.append({
                        "idx": idx,
                        "prompt": prompt,
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
                    f"[ERROR] Failed to generate valid output after 5 iterations.")
                return {
                    "spo_list_pred": [],
                    "history": history.copy(),
                    "final_output": llm_output,
                    "errorCode": NO_VALID_RESULT_WITHIN_MAX_RETRY,
                }

            history.append(f"Thought: {thought}")
            if debug:
                self.logger.info(f"Thought: {thought}")
            if action_name == "Finish":
                err_code, spo_list = self.parse_llm_output(args)

                finish_output = json.dumps(args, ensure_ascii=False)
                history.append(f"Finish: {finish_output}")
                if debug:
                    self.logger.info(f"Finish: {finish_output}")
                return {
                    "spo_list_pred": spo_list,
                    "history": history.copy(),
                    "final_output": llm_output,
                    "errorCode": err_code,
                }
            else:
                observation = self.tools[action_name].call(args)
                history.append(f"Action: {action_name}({args})")
                history.append(f"Observation: {observation}")
                if debug:
                    self.logger.info(f"Action: {action_name}({args})")
                    self.logger.info(f"Observation: {observation}")
        else:
            self.logger.error(
                f"[ERROR] Failed to generate valid output after 5 iterations.")
            return {
                "spo_list_pred": [],
                "history": history.copy(),
                "final_output": llm_output,
                "errorCode": NO_RESULT_WITHIN_MAX_ITERATIONS,
            }

    # @log_exceptions
    def train_sample(self, sample, idx):
        if configs['llm']['code_version'] == 'AgentRE':
            err_code = SUCCESS
            spo_list_pred = []
            summary_str = ""
            self.history = []

            text_str = json.dumps(sample['text'].strip(), ensure_ascii=False)
            if self.debug:
                self.logger.info(f"[idx={idx}] Input: {text_str}")

            # Outer loop: ReAct-fashion action search with max_iterations limit
            for _ in range(self.max_iterations):
                # [0] prompt
                prompt = self.generate_prompt(text_str)
                # if idx < 5: self.log_prompt(prompt)
                # Inner loop: try single action with max_retry limit
                err_code_, parsed_res = self.get_single_step(prompt, idx)
                if err_code != 0:
                    err_code = err_code_
                    break
                thought, action_name, args = parsed_res

                # [1] thought
                self.history.append(f"Thought: {thought}")
                if self.debug:
                    self.logger.info(f"Thought: {thought}")

                # [2] action
                if action_name == "Finish":
                    err_code, spo_list_pred = self.parse_llm_output(
                        json.loads(args))
                    if err_code < 0:
                        self.logger.error(
                            f"[ERROR] error in parse_llm_output: {args}")
                    self.history.append(f"Finish: {args}")
                    if self.debug:
                        self.logger.info(f"Finish: {args}")
                    # NEW: add refexion!
                    f1 = self.get_eval_result(sample['spo_list'], spo_list_pred)
                    if f1 < 1.0:
                        reflexion_sample = self.get_reflexion(
                            text_str, sample['spo_list'], spo_list_pred)
                        reflexion_text = json.dumps(
                            reflexion_sample, ensure_ascii=False)
                        self.history.append(f"Reflexion: {reflexion_text}")
                        if self.debug:
                            self.logger.info(f"Reflexion: {reflexion_text}")
                    else:
                        pass  
                    # NOTE: break when Finsh?
                    break
                else:
                    # If not "Finish", exec and generate observation
                    if action_name is None:
                        self.logger.warn("No action detected")
                        continue
                    observation = self.tools[action_name].call(args)
                    self.history.append(f"Action: {action_name}({args})")
                    self.history.append(f"Observation: {observation}")
                    if self.debug:
                        self.logger.info(f"Action: {action_name}({args})")
                        self.logger.info(f"Observation: {observation}")
            else:
                err_code = NO_RESULT_WITHIN_MAX_ITERATIONS
                self.logger.error(
                    f"[ERROR] Failed to generate valid output after 5 iterations.")

            # NEW: record into CorrectMemory!
            self.record_correct_memory(sample)

            # NEW: summary!
            if self.use_summary:
                summary_str = self.get_summary(
                    text_str, sample['spo_list'], self.history)
                self.history.append(f"Summary: {summary_str}")
                if self.debug:
                    self.logger.info(f"Summary: {summary_str}")
            return {
                "spo_list_pred": spo_list_pred,
                "history": self.history.copy(),
                "summary": summary_str,
                "errorCode": err_code,
            }
        elif configs['llm']['code_version'] == 'AgentRE_redux': 
            err_code = SUCCESS
            spo_list_pred = []
            summary_str = ""
            self.history = []

            text_str = json.dumps(sample['text'].strip(), ensure_ascii=False)
            if self.debug:
                self.logger.info(f"[idx={idx}] Input: {text_str}")

            correct_triples = []
            incorrect_triples = []

            # Outer loop: ReAct-fashion action search with max_iterations limit
            for _ in range(self.max_iterations):
                # [0] prompt
                prompt = self.generate_prompt(text_str)
                # if idx < 5: self.log_prompt(prompt)
                # Inner loop: try single action with max_retry limit
                err_code_, parsed_res = self.get_single_step(prompt, idx)
                if err_code != 0:
                    err_code = err_code_
                    break
                thought, action_name, args = parsed_res

                # [1] thought
                self.history.append(f"Thought: {thought}")
                if self.debug:
                    self.logger.info(f"Thought: {thought}")

                # [2] action
                if action_name == "Finish":
                    err_code, spo_list_pred = self.parse_llm_output(
                        json.loads(args))
                    if err_code < 0:
                        self.logger.error(
                            f"[ERROR] error in parse_llm_output: {args}")
                    self.history.append(f"Finish: {args}")
                    if self.debug:
                        self.logger.info(f"Finish: {args}")
                    # NEW: add refexion!
                    f1 = self.get_eval_result(sample['spo_list'], spo_list_pred)
                    if f1 < 1.0:
                        correct_triples, incorrect_triples = self.classify_triples(sample['spo_list'], spo_list_pred)
                        reflexion_sample = self.get_reflexion(
                            text_str, sample['spo_list'], incorrect_triples)
                        reflexion_text = json.dumps(
                            reflexion_sample, ensure_ascii=False)
                        self.history.append(f"Reflexion: {reflexion_text}")
                        if self.debug:
                            self.logger.info(f"Reflexion: {reflexion_text}")
                        self.record_reflexion_memory(reflexion_sample)
                    else:
                        pass 
                    # NOTE: break when Finsh?
                    break
                else:
                    # If not "Finish", exec and generate observation
                    if action_name is None:
                        self.logger.warn("No action detected")
                        continue
                    observation = self.tools[action_name].call(args)
                    self.history.append(f"Action: {action_name}({args})")
                    self.history.append(f"Observation: {observation}")
                    if self.debug:
                        self.logger.info(f"Action: {action_name}({args})")
                        self.logger.info(f"Observation: {observation}")
            else:
                err_code = NO_RESULT_WITHIN_MAX_ITERATIONS
                self.logger.error(
                    f"[ERROR] Failed to generate valid output after 5 iterations.")

            # NEW: record into CorrectMemory!
            self.record_correct_memory_v2(sample, correct_triples)
            self.record_incorrect_memory_v2(sample, incorrect_triples)

            # NEW: summary!
            if self.use_summary:
                summary_str = self.get_summary(
                    text_str, sample['spo_list'], self.history)
                self.history.append(f"Summary: {summary_str}")
                if self.debug:
                    self.logger.info(f"Summary: {summary_str}")
            return {
                "spo_list_pred": spo_list_pred,
                "history": self.history.copy(),
                "summary": summary_str,
                "errorCode": err_code,
            } 

    def get_single_step(self, prompt, idx):
        """ Try get single step action with self.max_retry
        return: err_code, (thought, action_name, args)
        """
        for _ in range(self.max_retry):
            llm_output = self.query_llm(
                prompt, stop=self.stop, temperature=0)

            if configs['data']['input_trace']:
                self.llm_inputs.append({
                    "idx": idx,
                    "prompt": prompt,
                    "llm_output": llm_output,
                })

            # 1. parse the output
            err_code, parsed_res = self.parse_output(llm_output)
            if err_code == -1:
                if self.debug:
                    self.logger.error(f"error in parse_output: {llm_output}")
                continue
            # 2. Action need to be valid
            thought, action_name, args = parsed_res
            if action_name not in self.tools:
                if self.debug:
                    self.logger.error(
                        f"error action_name: {action_name}. llm_output: {llm_output}")
                continue
            # 3. if "Finish", try to parse LLM output
            if action_name == "Finish":
                err_code, spo_list_pred = self.parse_llm_output(args)
                if err_code == -1:
                    if self.debug:
                        self.logger.error(
                            f"error in parse_llm_output: {args}. llm_output: {llm_output}")
                    continue
            return 0, (thought, action_name, json.dumps(args, ensure_ascii=False))
        else:
            self.logger.error(
                f"[ERROR] Failed to generate valid output after 5 iterations.")
            return NO_VALID_RESULT_WITHIN_MAX_RETRY, (None, None, None)

    def record_correct_memory(self, sample):
        """
        sample: {text, spo_list} """
        if isinstance(sample, list):
            index_texts = [format_sample_str(s) for s in sample]
            self.data_handler.correct_memory.add(index_texts)
        elif isinstance(sample, dict):
            index_text = format_sample_str(sample)
            self.data_handler.correct_memory.add(index_text)
        else:
            raise Exception(f"Unknown sample type: {type(sample)}")

    def record_incorrect_memory(self, sample, incorrect_triples):
        """
        sample: {text, incorrect_spo_list} """
        if isinstance(sample, list):
            index_texts = [format_incorrect_sample_str(s) for s in sample]
            self.data_handler.incorrect_memory.add(index_texts)
        elif isinstance(sample, dict):
            index_text = format_incorrect_sample_str(sample)
            self.data_handler.incorrect_memory.add(index_text)
        else:
            raise Exception(f"Unknown sample type: {type(sample)}")

    def get_reflexion(self, text, golden, pred):
        """ {text, golden, incorrect_triples/spo_list_pred} """
        prompt = self.prompter.get_reflexion_prompt(text, golden, pred)
        llm_output = self.query_llm(
            prompt, stop=self.stop, temperature=0).strip()
        reflexion = {
            "text": text,
            "golden": golden,
            "pred": pred,
            "reflexion": llm_output,
        }
        return reflexion  # json.dumps(reflexion, ensure_ascii=False)

    def record_reflexion_memory(self, reflexion_sample):
        """
        sample: {text, golden, pred, reflexion} """
        if isinstance(reflexion_sample, list):
            index_texts = [format_reflexion_sample_str(
                s) for s in reflexion_sample]
            self.data_handler.reflexion_memory.add(index_texts)
        elif isinstance(reflexion_sample, dict):
            index_text = format_reflexion_sample_str(reflexion_sample)
            self.data_handler.reflexion_memory.add(index_text)
        else:
            raise Exception(
                f"Unknown reflexion sample type: {type(reflexion_sample)}")

    # ======================== v2 methods ========================

    def classify_triples(self, golden_list: list[dict], pred_list: list[dict]):
        """Split predicted triples into correct and incorrect sets.
        
        Compares triples as JSON dicts (via json.dumps for normalization)
        without converting them to flat strings.
        
        Returns:
            correct_triples:   list[dict] -- predicted triples that match a golden triple (TP)
            incorrect_triples: dict with 'false_positives' and 'false_negatives'
        """
        # Build lookup: normalized JSON string -> original dict
        golden_norm = {json.dumps(t, sort_keys=True, ensure_ascii=False): t for t in golden_list}
        pred_norm = {json.dumps(t, sort_keys=True, ensure_ascii=False): t for t in pred_list}

        golden_keys = set(golden_norm.keys())
        pred_keys = set(pred_norm.keys())

        tp_keys = golden_keys & pred_keys
        fp_keys = pred_keys - golden_keys
        fn_keys = golden_keys - pred_keys

        correct_triples = [golden_norm[k] for k in tp_keys]
        incorrect_triples = []
        incorrect_triples.extend([pred_norm[k] for k in fp_keys])
        incorrect_triples.extend([golden_norm[k] for k in fn_keys])

        return correct_triples, incorrect_triples

    def record_correct_memory_v2(self, sample, correct_triples: list[dict]):
        """Record only the correctly predicted triples (TP) into CorrectMemory.
        
        Unlike record_correct_memory which stores ALL ground-truth triples,
        this only stores the triples the model got right.
        
        Args:
            sample: dict with 'text' key
            correct_triples: list of triple dicts that were correctly predicted
        """
        if len(correct_triples) == 0:
            return
        partial_sample = {
            'text': sample['text'],
            'spo_list': correct_triples,
        }
        index_text = format_sample_str(partial_sample)
        self.data_handler.correct_memory.add(index_text)

    def record_incorrect_memory_v2(self, sample, incorrect_triples: list[dict]):
        """Record only the incorrectly predicted triples (FP+FN) into IncorrectMemory.
        
        Args:
            sample: dict with 'text' key
            incorrect_triples: list of triple dicts that were incorrectly predicted
        """
        if len(incorrect_triples) == 0:
            return
        partial_sample = {
            'text': sample['text'],
            'incorrect_spo_list': incorrect_triples,
        }
        index_text = format_incorrect_sample_str(partial_sample)
        self.data_handler.incorrect_memory.add(index_text)

    def record_reflexion_memory_v2(self, text, golden: list[dict], incorrect_triples: dict):
        """Generate and record reflexion only on incorrect triples.
        
        Args:
            text: the input text string
            golden: the golden triples
            incorrect_triples: dict with 'false_positives' and 'false_negatives'
        
        Returns:
            reflexion_sample: the generated reflexion dict
        """
        reflexion_sample = self.get_reflexion(text, golden, incorrect_triples)
        self.record_reflexion_memory(reflexion_sample)
        return reflexion_sample

    # ==================== end v2 methods ====================

    def get_summary(self, text, golden, history):
        """ {text, golden, history} """
        prompt = self.prompter.get_summary_prompt(text, golden, history)
        llm_output = self.query_llm(
            prompt, stop=self.stop, temperature=0).strip()
        return json.dumps(llm_output, ensure_ascii=False)

    def get_eval_result(self, golden, pred):
        """
        return:
            triplet_correct, triplet_wrong ?
        """
        self.evaluator.add(golden, pred)
        last_TP, last_FN, last_FP = self.evaluator.get_last_metric()
        f1 = round(last_TP / (last_TP + 0.5 * (last_FP + last_FN)), 4)
        # precision = round(last_TP / (last_TP + last_FP), 4)
        # recall = round(last_TP / (last_TP + last_FN), 4)
        # f1 = round(2 * precision * recall / (precision + recall), 4)
        return f1

    def generate_prompt(self, text, inference=False):
        if configs['llm']['code_version'] == 'AgentRE':
            tools_desc = "\n".join(
                [f"- {tool.name}: {tool.get_description()}" for tool in self.tools.values()])
            task_description = self.tools['GetTaskDescription'].call()
            retrieved_examples = self.tools['RetrieveCorrectMemory'].call(text)
            prompt = self.prompter.get_react_prompt(text, tools_desc) + \
                self.prompter.get_react_first_step(task_description) + \
                self.prompter.get_react_second_step(text, retrieved_examples)
            if len(self.history) == 0:
                self.history.append(f"Action: GetTaskDescription()")
                self.history.append(f"Observation: {task_description}")
                self.history.append(f"Action: RetrieveCorrectMemory({text})")
                self.history.append(f"Observation: {retrieved_examples}")
                if self.debug:
                    self.logger.info(f"Action: GetTaskDescription()")
                    self.logger.info(f"Observation: {task_description}")
                    self.logger.info(f"Action: RetrieveCorrectMemory({text})")
                    self.logger.info(f"Observation: {retrieved_examples}")
            # Action+Observation
            for history in self.history[self.num_pre_history * 2:]:
                prompt += history + "\n"
            prompt += self.prompter.get_react_suffix()
        elif configs['llm']['code_version'] == 'AgentRE_redux':
            tools_desc = "\n".join(
                [f"- {tool.name}: {tool.get_description()}" for tool in self.tools.values()])
            task_description = self.tools['GetTaskDescription'].call()
            retrieved_examples = self.tools['RetrieveCorrectMemory'].call(text)
            entity_info = self.tools['RetrieveRelevantInfo'].call(text)
            retrieved_incorrect_examples = self.tools['RetrieveIncorrectMemory'].call(text)
            prompt = self.prompter.get_react_prompt(text, tools_desc) + \
                self.prompter.get_react_first_step(task_description) + \
                self.prompter.get_react_second_step(
                    text, retrieved_examples) + \
                self.prompter.get_entity_info_step(text, entity_info) + \
                self.prompter.get_incorrect_memory_step(text, retrieved_incorrect_examples)
            if len(self.history) == 0:
                self.history.append(f"Action: GetTaskDescription()")
                self.history.append(f"Observation: {task_description}")
                self.history.append(f"Action: RetrieveCorrectMemory({text})")
                self.history.append(f"Observation: {retrieved_examples}")
                self.history.append(f"Action: RetrieveRelevantInfo({text})")
                self.history.append(f"Observation: {entity_info}")
                self.history.append(f"Action: RetrieveIncorrectMemory({text})")
                self.history.append(f"Observation: {retrieved_incorrect_examples}")
                if self.debug:
                    self.logger.info(f"Action: GetTaskDescription()")
                    self.logger.info(f"Observation: {task_description}")
                    self.logger.info(f"Action: RetrieveCorrectMemory({text})")
                    self.logger.info(f"Observation: {retrieved_examples}")
                    self.logger.info(f"Action: RetrieveRelevantInfo({text})")
                    self.logger.info(f"Observation: {entity_info}")
                    self.logger.info(f"Action: RetrieveIncorrectMemory({text})")
                    self.logger.info(f"Observation: {retrieved_incorrect_examples}")
            # Action+Observation
            # everything prior to self.num_pre_history * 2 was just added in the previous len(self.history) == 0 block
            if len(self.history) > self.num_pre_history * 2:
                if inference: 
                    # No reflexions for inference so its not in the history by default
                    # We retrieve from the reflexion memory to add to the prompt.
                    retrieved_reflexion_examples = self.tools['RetrieveReflexionMemory'].call(
                        text)
                    prompt += self.prompter.get_reflexion_step(
                        text, retrieved_reflexion_examples)
                for hist_event in self.history[self.num_pre_history * 2:]:
                    prompt += str(hist_event) + "\n"
                prompt += "\n"
            prompt += self.prompter.get_react_suffix()
        return prompt

    def parse_output(self, llm_output: str):
        try:
            # regex = r"(.*?)\nAction:(.*?)\nActionInput:[\s]*(.*)"
            react_regex = r"(.*?)Action:(.*?)\nActionInput:[\s]*(.*)"
            result_regex = r"(.*?)spo_list(.*))"
            match = re.search(react_regex, llm_output, re.DOTALL)
            if match:
                thought = match.group(1).strip()
                action = match.group(2).strip()
                args = match.group(3).strip()
                thought = json.dumps(thought, ensure_ascii=False)
                return 0, (thought, action, args)
            else:
                return 0, ("", "Finish", llm_output)

        except Exception as e:
            return -1, None
