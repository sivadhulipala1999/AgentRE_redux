import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def adaptive_chunk(text: str, 
                   sim_threshold: float = 0.5, 
                   max_words: int = 500, 
                   min_words: int = 80) -> list[str]:
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    embeddings = embedder.encode(sentences)
    
    chunks = []
    current_chunk = [sentences[0]]
    current_words = len(sentences[0].split())
    
    for i in range(1, len(sentences)):
        sim = cosine_similarity(
            [embeddings[i]], [embeddings[i-1]]
        )[0][0]
        new_words = len(sentences[i].split())

        would_exceed_max = (current_words + new_words) > max_words
        too_dissimilar = sim < sim_threshold
        below_min = current_words < min_words
        
        if (too_dissimilar or would_exceed_max) and not below_min:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
            current_words = new_words
        else:
            current_chunk.append(sentences[i])
            current_words += new_words
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks