
from modules.tools import GetTaskDescription
from logging import getLogger
from data_utils.data_handler_re import DataHandlerRE
from config.configurator import configs
from .prompt_zh import *
from .prompt_en import *
from .prompt_scagent_en import *
import json
from typing import List


class BasePrompter:
    def __init__(self, data_handler):
        self.data_handler: DataHandlerRE = data_handler
        self.logger = getLogger('train_logger')
        self.language = data_handler.data_meta.language

class PrompterSCAgent(BasePrompter):
    def __init__(self, data_handler):
        super().__init__(data_handler)
        if self.language == "en":
            self.SYSTEM_PROMPT = SYSTEM_PROMPT 
            self.SYSTEM_PROMPT_C = SYSTEM_PROMPT_C
            self.USER_PROMPT_A = USER_PROMPT_A
            self.USER_PROMPT_B = USER_PROMPT_B
            self.USER_PROMPT = USER_PROMPT 
            self.RETRIEVER_OUTPUT_TEMPLATE = RETRIEVER_OUTPUT_TEMPLATE
        else:
            raise ValueError(f"Unsupported language: {self.language}")

    def get_react_prompt(self, text: str, tools_desc: str):
        return self.TEMPLATE_REACT.format(tools=tools_desc, text=text)
    
    def get_system_prompt(self):
        return self.SYSTEM_PROMPT # Complete setup

    def get_system_prompt_c(self): 
        return self.SYSTEM_PROMPT_C # Flat FSL, no staged detection, let the model decide
    
    def format_retrieved_examples(self, retrieved_examples: List[dict]):
        output = "" 
        for idx, example in enumerate(retrieved_examples):
            output += "\n\n" + self.RETRIEVER_OUTPUT_TEMPLATE.format(i=idx+1, text=example['text'], spo_list=example['spo_list'])
        return output

    def get_user_prompt(self, retrieved_examples: str, chunk_text: str):
        return self.USER_PROMPT.format(retrieved_examples=retrieved_examples, chunk_text=chunk_text)
    
    def get_static_user_prompt(self, static_example: str, chunk_text: str):
        return self.USER_PROMPT_B.format(static_example=static_example, chunk_text=chunk_text)
    
    def get_static_user_prompt_no_examples(self, chunk_text: str):
        return self.USER_PROMPT_A.format(chunk_text=chunk_text)


class PrompterReActFSL(BasePrompter):
    def __init__(self, data_handler):
        super().__init__(data_handler)
        if self.language == "zh":
            self.TEMPLATE_REFLEXION = TEMPLATE_REFLEXION_ZH
            self.TEMPLATE_REACT = TEMPLATE_REACT_ZH
            self.FIRST_STEP = FIRST_STEP_ZH
            self.SECOND_STEP = SECOND_STEP_ZH
        elif self.language == "en":
            self.TEMPLATE_REFLEXION = TEMPLATE_REFLEXION_EN
            self.TEMPLATE_REACT = TEMPLATE_REACT_EN
            self.FIRST_STEP = FIRST_STEP_EN
            self.SECOND_STEP = SECOND_STEP_EN
        else:
            raise ValueError(f"Unsupported language: {self.language}")
        self.SUFFIX = SUFFIX

    def get_react_prompt(self, text: str, tools_desc: str):
        return self.TEMPLATE_REACT.format(tools=tools_desc, text=text)

    def get_react_first_step(self, task_description: str):
        return self.FIRST_STEP.format(task_description=task_description)

    def get_react_second_step(self, text: str, retrieved_examples: str):
        return self.SECOND_STEP.format(text=text, retrieved_examples=retrieved_examples)

    def get_react_suffix(self):
        return SUFFIX


class PrompterReActMemory(BasePrompter):
    """ 相较于 PrompterReActFSL
    - second step 中的召回接口不同
    - 增加 Refelxion
    """

    def __init__(self, data_handler):
        super().__init__(data_handler)
        if self.language == "zh":
            self.TEMPLATE_REFLEXION = TEMPLATE_REFLEXION_ZH if configs['llm']['code_version'] == 'AgentRE' else TEMPLATE_REFLEXION_ZH_REDUX
            self.TEMPLATE_REACT = TEMPLATE_REACT_ZH
            self.FIRST_STEP = FIRST_STEP_ZH
            self.SECOND_STEP = SECOND_STEP_MEMORY_ZH if configs['llm']['code_version'] == 'AgentRE' else SECOND_STEP_MEMORY_ZH_REDUX
            self.TEMPLATE_SUMMAY = TEMPLATE_SUMMAY_ZH
            self.ENTITY_INFO_STEP = ENTITY_INFO_STEP_ZH
            self.REFLEXION_STEP = REFLEXION_STEP_ZH if configs['llm']['code_version'] == 'AgentRE' else REFLEXION_STEP_ZH_REDUX
            self.INCORRECT_MEMORY_STEP = INCORRECT_MEMORY_STEP_ZH if configs['llm']['code_version'] == 'AgentRE_redux' else ''
        elif self.language == "en":
            self.TEMPLATE_REFLEXION = TEMPLATE_REFLEXION_EN if configs['llm']['code_version'] == 'AgentRE' else TEMPLATE_REFLEXION_EN_REDUX
            self.TEMPLATE_REACT = TEMPLATE_REACT_EN
            self.FIRST_STEP = FIRST_STEP_EN
            self.SECOND_STEP = SECOND_STEP_MEMORY_EN if configs['llm']['code_version'] == 'AgentRE' else SECOND_STEP_MEMORY_EN_REDUX
            self.TEMPLATE_SUMMAY = TEMPLATE_SUMMAY_EN
            self.ENTITY_INFO_STEP = ENTITY_INFO_STEP_EN
            self.REFLEXION_STEP = REFLEXION_STEP_EN if configs['llm']['code_version'] == 'AgentRE' else REFLEXION_STEP_EN_REDUX
            self.INCORRECT_MEMORY_STEP = INCORRECT_MEMORY_STEP_EN if configs['llm']['code_version'] == 'AgentRE_redux' else ''
        else:
            raise ValueError(f"Unsupported language: {self.language}")
        self.SUFFIX = SUFFIX

    def get_react_prompt(self, text: str, tools_desc: str):
        return self.TEMPLATE_REACT.format(tools=tools_desc, text=text)

    def get_react_first_step(self, task_description: str):
        return self.FIRST_STEP.format(task_description=task_description)

    def get_react_second_step(self, text: str, retrieved_examples: str):
        return self.SECOND_STEP.format(text=text, retrieved_examples=retrieved_examples)

    def get_reflexion_step(self, text: str, retrieved_reflexion_samples: str):
        return self.REFLEXION_STEP.format(text=text, retrieved_examples=retrieved_reflexion_samples)

    def get_entity_info_step(self, text: str, entity_info: str):
        return self.ENTITY_INFO_STEP.format(text=text, entity_info=entity_info)
    
    def get_incorrect_memory_step(self, text: str, retrieved_incorrect_examples: str):
        return self.INCORRECT_MEMORY_STEP.format(text=text, retrieved_examples=retrieved_incorrect_examples)

    def get_react_suffix(self):
        return SUFFIX

    def get_reflexion_prompt(self, text: str, golden: str, pred: str):
        golden, pred = json.dumps(golden, ensure_ascii=False), json.dumps(
            pred, ensure_ascii=False)
        return self.TEMPLATE_REFLEXION.format(text=text, golden=golden, pred=pred)

    def get_summary_prompt(self, text: str, golden: str, history: List[str]):
        if isinstance(golden, list):
            golden = json.dumps(golden, ensure_ascii=False)
        history = "\n".join(history)
        return self.TEMPLATE_SUMMAY.format(text=text, golden=golden, history=history)
