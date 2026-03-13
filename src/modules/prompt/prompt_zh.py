TEMPLATE_REACT_ZH = \
    """尽你所能, 从给定的句子中识别出符合规范格式要求的关系三元组. 

你可以使用以下的工具: 
{tools}

使用如下的形式进行回答:
Thought: 思考下一步应该做什么
Action: 执行的动作名称, 需要在上面的工具列表中
ActionInput: 执行的动作所传入的一个参数, 可以为空
Observation: 执行动作后返回的结果
... (上面的 Thought/Action/ActionInput/Observation 三个步骤可以重复多次, 直到执行Finish动作返回结果)

Begin! 
输入的句子是 `{text}`\n
"""
FIRST_STEP_ZH = \
    """Thought: 首先，我需要了解更多关于关系三元组抽取任务的定义和输出格式的信息。
Action: GetTaskDescription
ActionInput:
Observation: {task_description}\n"""
SECOND_STEP_ZH = \
    """Thought: 我可以先观察一些已经标注好的关系三元组，以便更好地理解这个任务。
Action: RetrieveExamples
ActionInput: {text}
Observation: {retrieved_examples}\n"""
SUFFIX = """Thought: """

SECOND_STEP_MEMORY_ZH = \
    """Thought: 我可以找到已有的正确的例子来帮助我理解这个任务。
Action: RetrieveCorrectMemory
ActionInput: {text}
Observation: {retrieved_examples}\n"""

SECOND_STEP_MEMORY_ZH_REDUX = \
    """Thought: 我可以从现有的正确示例中找到一些例子来帮助我理解这项任务。但必须指出，这些示例可能仅代表文本中全部三元组的子集，因为它们是从模型的先前预测中筛选出来的。
Action: RetrieveCorrectMemory
ActionInput: {text}
Observation: {retrieved_examples}\n"""

TEMPLATE_REFLEXION_ZH = \
    """在关系抽取任务中, 对于输入的句子 `{text}`, 正确的结果应该是 `{golden}`. 但模型输出的结果是 `{pred}`. 
请你用一句话来总结错误的原因: """

TEMPLATE_REFLEXION_ZH_REDUX = \
    """在关系抽取任务中, 对于输入的句子 `{text}`, 正确的结果应该是 `{golden}`. 但该模型的预测结果与实际情况不符，其中预测错误的三元组为 `{pred}`.
请你用一句话来总结错误的原因: """

TEMPLATE_SUMMAY_ZH = \
    """在关系抽取任务中, 对于输入的句子 `{text}`, 正确的结果应该是 `{golden}`. 下面是可以参考的抽取过程: 
```
{history}
```
假如你无法在抽取过程中执行这些 Action, 需要直接生成抽取结果, 请用一句话给出你的推理依据, 并给出最终的JSON抽取结果: """

ENTITY_INFO_STEP_ZH = \
    """Thought: 我可以提取关于实体的信息，以进一步改进我的预测。格式如下: 'entity---entity_info'.

Action: RetrieveRelevantInfo
ActionInput: {text}         
Observation: {entity_info}\n"""

REFLEXION_STEP_ZH = \
    """Thought: 我可以回顾之前对给定文本的思考，以提高我的预测能力。
Action: RetrieveReflexionMemory         
ActionInput: {text}
Observation: {retrieved_examples}\n"""

REFLEXION_STEP_ZH_REDUX = \
    """Thought: 我可以回顾自己对该文本先前错误预测的反思，以此改进当前的预测。
Action: RetrieveReflexionMemory
ActionInput: {text}
Observation: {retrieved_examples}\n"""


INCORRECT_MEMORY_STEP_ZH = \
    """Thought: 我能够找出在先前步骤中错误预测的三元组，以此来修正当前的预测结果。
Action: RetrieveIncorrectMemory
ActionInput: {text}
Observation: {retrieved_examples}\n"""