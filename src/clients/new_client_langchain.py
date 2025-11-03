import sys
from typing import List, Dict, Any

# LangChain components
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache


from config.configurator import configs
from diskcache import Cache
from openai import OpenAI

cache = Cache('./cache/llama.cache')

GENERATIVE_MODELS = ["gpt-3.5-turbo-instruct"]

MODELS = ["gpt-3.5-turbo-instruct", "meta-llama-3.1-8b-instruct"]

# --- Helper functions to bridge your dicts with LangChain messages ---


def _dicts_to_messages(history_dicts: List[Dict[str, str]]) -> List[BaseMessage]:
    """Converts a list of message dicts to a list of LangChain BaseMessage objects."""
    messages = []
    for msg in history_dicts:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        elif role == "system":
            messages.append(SystemMessage(content=content))
    return messages


def _get_content(response: Any) -> str:
    """Extracts the string content from a LangChain model response."""
    if isinstance(response, BaseMessage):
        return response.content
    elif isinstance(response, str):
        return response
    return str(response)

# --- Your Class, Refactored ---
# (Assuming 'configs', 'API_KEYS', 'GENERATIVE_MODELS', 'cache' are defined elsewhere)


class OpenAIClient:
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 4096

    history: list = []  # Still stores dicts, as per your original class

    use_cache: bool = configs['llm'].get('use_cache', False)

    def __init__(self, model_name: str = None, temperature: float = None, max_tokens: int = None):
        if model_name:
            self.model_name = model_name
        if temperature:
            self.temperature = temperature
        if max_tokens:
            self.max_tokens = max_tokens

        self.is_generative_model = self.model_name in GENERATIVE_MODELS

        # 1. Set up LangChain's global cache
        if self.use_cache:
            set_llm_cache(InMemoryCache())
            # We no longer need the manual 'cache' object from the original query_one

        # 2. Determine client arguments for LangChain
        client_kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if configs['llm']['client'] == 'academic':
            client_kwargs["base_url"] = "https://chat-ai.academiccloud.de/v1"
            client_kwargs["api_key"] = API_KEYS[1]
        else:
            client_kwargs["base_url"] = None
            client_kwargs["api_key"] = API_KEYS[0]

        # 3. Instantiate the correct LangChain model and store it
        if self.is_generative_model:
            # Use the legacy 'OpenAI' class for generative models
            self.llm = OpenAI(model=self.model_name, **client_kwargs)
        else:
            # Use 'ChatOpenAI' for all chat-based models
            self.llm = ChatOpenAI(model_name=self.model_name, **client_kwargs)

        # This replaces self.client
        # Note: self.client is no longer used, but we've initialized self.llm

    def query_chat(self, text, stop=None, temperature=None) -> str:
        """Internal query for chat models (single turn)."""
        # if self.is_generative_model:
        #     raise TypeError("Cannot use query_chat with a generative model.")

        kwargs = {
            "temperature": self.temperature if temperature is None else temperature,
            "stop": stop
        }
        # Convert the single message dict to a LangChain message list
        messages = [HumanMessage(content=text)]

        # Use LangChain's .invoke() method
        response = self.llm.invoke(messages, **kwargs)
        return _get_content(response)

    def query_generative(self, text, stop=None, temperature=None) -> str:
        """Internal query for legacy generative models."""
        if not self.is_generative_model:
            raise TypeError("Cannot use query_generative with a chat model.")

        kwargs = {
            "temperature": self.temperature if temperature is None else temperature,
            "stop": stop
        }

        # Legacy models just take a string
        response = self.llm.invoke(text, **kwargs)
        return _get_content(response).strip()

    def query_one(self, text, stop=None, temperature=None) -> str:
        """
        Query with a single string. Caching is now automatic via LangChain.
        """
        # The original cache logic is no longer needed.
        # LangChain's global cache (set in __init__) handles this automatically.

        if self.is_generative_model:
            res = self.query_generative(
                text, stop=stop, temperature=temperature)
        else:
            res = self.query_chat(text, stop=stop, temperature=temperature)

        # The original cache.set() is also no longer needed.
        return res

    def query_one_stream(self, text) -> None:
        """ Just for test! Streams the response. """
        if self.is_generative_model:
            raise TypeError(
                "Streaming is only implemented for chat models here.")

        kwargs = {
            "temperature": self.temperature,
            "stop": ["\nObservation:"]
        }

        messages = [HumanMessage(content=text)]

        # Use LangChain's .stream() method
        stream = self.llm.stream(messages, **kwargs)
        for chunk in stream:
            print(_get_content(chunk) or "", end="")
        print()  # Add a newline at the end

    def chat(self, text, stop=None, temperature=None) -> str:
        """Stateful chat that uses the internal self.history."""
        if self.is_generative_model:
            raise TypeError(
                "Stateful .chat() is not supported for generative models.")

        # 1. Append the user's dict message to history
        self.history.append({"role": "user", "content": text})

        kwargs = {
            "temperature": self.temperature if temperature is None else temperature,
            "stop": stop
        }

        # 2. Convert the *entire* dict history to LangChain messages for the API call
        messages = _dicts_to_messages(self.history)

        # 3. Call .invoke()
        response = self.llm.invoke(messages, **kwargs)
        res_content = _get_content(response)

        # 4. Append the assistant's dict message back to history
        self.history.append({"role": "assistant", "content": res_content})
        return res_content

    def clear_history(self):
        """Clears the internal conversation history."""
        self.history = []

    def chat_with_history(self, history: list, stop=None, temperature=None) -> str:
        """Stateless chat that takes an external history list."""
        if self.is_generative_model:
            raise TypeError(
                ".chat_with_history() is not supported for generative models.")

        kwargs = {
            "temperature": self.temperature if temperature is None else temperature,
            "stop": stop
        }

        # 1. Convert the provided dict history to LangChain messages
        messages = _dicts_to_messages(history)

        # 2. Call .invoke()
        response = self.llm.invoke(messages, **kwargs)
        return _get_content(response)
