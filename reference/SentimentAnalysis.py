# coding: utf-8
# Upstream: PacktPublishing/Python-Artificial-Intelligence-Projects-for-Beginners
# Chapter03/SentimentAnalysis.py (book Doc2Vec exploration on labelled sentences)

import gensim
import logging
import re
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

DATA_DIR = "../data/sentiment_labelled_sentences"


def load_sentences():
    sentences = []
    sentiments = []
    for fname in ["yelp", "amazon_cells", "imdb"]:
        with open(f"{DATA_DIR}/{fname}_labelled.txt", encoding="utf-8") as f:
            for item_no, line in enumerate(f):
                line_split = line.strip().split("\t")
                sent = line_split[0].lower()
                sent = re.sub(r"'", "", sent)
                sent = re.sub(r"\W", " ", sent)
                sent = re.sub(r"\s+", " ", sent).strip()
                sentences.append(TaggedDocument(sent.split(), [f"{fname}_{item_no}"]))
                sentiments.append(int(line_split[1]))
    return sentences, sentiments


if __name__ == "__main__":
    sentences, _ = load_sentences()
    model = Doc2Vec(sentences, min_count=1, vector_size=50, epochs=30)
    print(model.most_similar("tasty"))
