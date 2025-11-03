from datasets import Dataset
import json
import spacy
import requests
import urllib
from modules.tools.base_tool import BaseTool
from config.configurator import configs
from modules.retrieval.index import DummyIndex, SimCSEIndex, BGEIndex, BaseIndex, MODE2INDEX
from modules.module_utils import format_sample, format_sample_str
from logging import getLogger
logger = getLogger('train_logger')


class RetrieveRelevantInfo(BaseTool):
    name: str = "RetrieveRelevantInfo"
    description_en: str = "Retrieve relevant information to the user input. The input is a sentence. "
    description_zh: str = "檢索與使用者輸入相關的資訊。輸入是一個句子。"

    def init(self):
        self.entity_recognizer = spacy.load("en_core_web_sm")
        logger.info("RetrieveRelevantInfo: Entity Recognizer initialized")

    def extract_entities(self, text):
        doc = self.entity_recognizer(text)
        return list({ent.text for ent in doc.ents})

    def wiki_summary(self, entity):
        url = (
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(entity)}"
        )
        res = requests.get(
            url, headers={"User-Agent": "AgentREEntityLookup/0.1"}, timeout=10).json()
        return res["extract"] if res.get("extract") else ''

    def call(self, query):
        try:
            entities = self.extract_entities(query)
            all_facts = []
            for ent in entities:
                all_facts.append(ent + "---" + self.wiki_summary(ent))
            return "\n\n".join(all_facts) if all_facts else "No facts found for extracted entities."
        except:
            logger.error("error in relevant_info_retrieval")
