from modules.tools.base_tool import BaseTool
from config.configurator import configs


class RetrieveCorrectMemory(BaseTool):
    """ see['memory']['CorrectMemory'] """
    name: str = "RetrieveCorrectMemory"
    description_en: str = "Retrieve examples from CorrectMemory which can help to judge. The input is a sentence. "
    description_zh: str = "从 CorrectMemory 中召回相似的例子来帮助判断. 传入参数是句子, 可以对于当前的句子改写作为输入. "

    def init(self):
        self.correct_memory = self.data_handler.correct_memory

    def call(self, query):
        return self.correct_memory.query(query)


class RetrieveReflexionMemory(BaseTool):
    """ see ['memory']['ReflexionMemory'] """
    name: str = "RetrieveReflexionMemory"
    description_en: str = "Retrieve examples from ReflexionMemory which can help to learn from wrong experiments. The input is a sentence. "
    description_zh: str = "从此前错误的抽取中召回相似的例子来帮助学习. 传入参数是句子, 可以对于当前的句子改写作为输入. "

    def init(self):
        self.reflexion_memory = self.data_handler.reflexion_memory

    def call(self, query):
        return self.reflexion_memory.query(query)


class RetrieveIncorrectMemory(BaseTool):
    """ see ['memory']['IncorrectMemory'] """
    name: str = "RetrieveIncorrectMemory"
    description_en: str = "Retrieve examples from IncorrectMemory which can help to learn from wrong experiments. The input is a sentence. "
    description_zh: str = "从IncorrectMemory中检索示例, 这些示例有助于从错误实验中学习。输入是一个句子。"

    def init(self):
        self.incorrect_memory = self.data_handler.incorrect_memory

    def call(self, query):
        return self.incorrect_memory.query(query)