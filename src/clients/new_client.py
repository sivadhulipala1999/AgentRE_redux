""" 
repo: https://github.com/openai/openai-python
API: https://platform.openai.com/docs/guides/text-generation
"""

from config.configurator import configs
from diskcache import Cache
from openai import OpenAI
cache = Cache('./cache/llama.cache')

GENERATIVE_MODELS = ["gpt-3.5-turbo-instruct"]

MODELS = ["gpt-3.5-turbo-instruct", "meta-llama-3.1-8b-instruct"]

# base_url = "https://chat-ai.academiccloud.de/v1"
# model = "meta-llama-3.1-8b-instruct"  # Choose any available model


class OpenAIClient:
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 4096

    history: list = []      # conversation history

    use_cache: bool = configs['llm']['use_cache'] if 'use_cache' in configs['llm'] else False
    # cache_key: str = configs['llm']['cache_key'] if 'cache_key' in configs['llm'] else 'default_cache_key'

    def __init__(self, model_name: str = None, temperature: float = None, max_tokens: int = None):
        if model_name:
            self.model_name = model_name
        if temperature:
            self.temperature = temperature
        if max_tokens:
            self.max_tokens = max_tokens
        self.is_generative_model = self.model_name in GENERATIVE_MODELS
        client_kwargs = dict()
        if configs['llm']['client'] == 'academic':
            client_kwargs["base_url"] = "https://chat-ai.academiccloud.de/v1"
            client_kwargs["api_key"] = API_KEYS[1]
        else:
            client_kwargs["api_key"] = API_KEYS[0]
        self.client = OpenAI(**client_kwargs)

    def query_chat(self, text, stop=None, temperature=None) -> str:
        kwargs = {"messages": [{"role": "user", "content": text, }],
                  "model": self.model_name,
                  "temperature": self.temperature if temperature is None else temperature,
                  "max_tokens": self.max_tokens,
                  "stop": stop}
        # if "llama" in self.model_name:
        #     kwargs["base_url"] = "https://chat-ai.academiccloud.de/v1"
        chat_completion = self.client.chat.completions.create(**kwargs)
        return chat_completion.choices[0].message.content

    def query_generative(self, text, stop=None, temperature=None) -> str:
        kwargs = {
            "prompt": text,
            "model": self.model_name,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
            "stop": stop
        }
        # if "llama" in self.model_name:
        #     kwargs["base_url"] = "https://chat-ai.academiccloud.de/v1"
        completion = self.client.completions.create(**kwargs)
        return completion.choices[0].text.strip()

    def query_one(self, text, stop=None, temperature=None) -> str:
        cache_key_ = f"{self.model_name}_{text}"
        if self.use_cache and cache_key_ in cache:
            return cache.get(cache_key_)
        else:
            if self.is_generative_model:
                res = self.query_generative(
                    text, stop=stop, temperature=temperature)
            else:
                res = self.query_chat(text, stop=stop, temperature=temperature)
            cache.set(cache_key_, res)
        return res

    def query_one_stream(self, text) -> None:
        """ Just for test! """
        kwargs = {
            "messages": [{"role": "user", "content": text, }],
            "model": self.model_name,
            "temperature": self.temperature,
            "stream": True,
            "stop": ["\nObservation:"]
        }
        # if "llama" in self.model_name:
        #     kwargs["base_url"] = "https://chat-ai.academiccloud.de/v1"
        stream = self.client.chat.completions.create(**kwargs)
        for chunk in stream:
            print(chunk.choices[0].delta.content or "", end="")

    def chat(self, text, stop=None, temperature=None) -> str:
        self.history.append({"role": "user", "content": text})
        kwargs = {
            "messages": self.history,
            "model": self.model_name,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
            "stop": stop
        }
        # if "llama" in self.model_name:
        #     kwargs["base_url"] = "https://chat-ai.academiccloud.de/v1"
        chat_completion = self.client.chat.completions.create(**kwargs)
        res = chat_completion.choices[0].message.content
        self.history.append({"role": "assistant", "content": res})
        return res

    def clear_history(self):
        self.history = []

    def chat_with_history(self, history: list, stop=None, temperature=None) -> str:
        kwargs = {
            "messages": history,
            "model": self.model_name,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
            "stop": stop
        }
        # if "llama" in self.model_name:
        #     kwargs["base_url"] = "https://chat-ai.academiccloud.de/v1"
        chat_completion = self.client.chat.completions.create(**kwargs)
        res = chat_completion.choices[0].message.content
        return res
