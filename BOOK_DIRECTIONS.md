# Book directions — User review sentiment

**Catalog #1188** · Joshua Eckroth, *Python Artificial Intelligence Projects for Beginners*, **Chapter 3** — “Detecting positive or negative sentiments in user reviews”

- Packt repo: [Chapter03/SentimentAnalysisnew.ipynb](https://github.com/PacktPublishing/Python-Artificial-Intelligence-Projects-for-Beginners/tree/master/Chapter03)
- Local reference: [`reference/SentimentAnalysis.py`](./reference/SentimentAnalysis.py) (upstream script)
- Shared logic: [`sentiment_core.py`](./sentiment_core.py)
- Labelled data (Kotzias et al., KDD 2015): [`data/sentiment_labelled_sentences/`](./data/sentiment_labelled_sentences/)

## What the book asks you to build

Classify **product / movie / restaurant** review sentences as **positive** or **negative** using:

1. **Text preprocessing** (lowercase, strip HTML/punctuation, tokenize).
2. **Doc2Vec** (`gensim`) to embed each sentence.
3. **k-nearest neighbors** (`sklearn`, k=9) on document vectors.

The book trains Doc2Vec on large unlabeled corpora; this demo trains on the **3,000 labelled sentences** (Yelp, Amazon, IMDb) so it runs locally without downloading ACL IMDb.

## Data format

Each line: `sentence<TAB>label` where label is `1` (positive) or `0` (negative).

## Web version (`web/` — FastAPI, port **3008**)

| Book step | Web equivalent |
|-----------|----------------|
| Preprocess + `infer_vector` | [`sentiment_core.extract_words`](./sentiment_core.py) + Doc2Vec |
| k-NN classify | `POST /api/predict` |
| Evaluate with cross-validation | Run once at train time; `GET /api/status` reports accuracy |
| Try sample reviews | Example chips in UI |

First startup trains and caches models under `web/data/cache/` (about one minute).
