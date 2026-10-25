# User review sentiment (catalog #1188)

Joshua Eckroth — *Python Artificial Intelligence Projects for Beginners*, Chapter 3.

| Artifact | Path |
|----------|------|
| Book steps | [`BOOK_DIRECTIONS.md`](./BOOK_DIRECTIONS.md) |
| Upstream script | [`reference/SentimentAnalysis.py`](./reference/SentimentAnalysis.py) |
| CLI | [`reference/run_cli.py`](./reference/run_cli.py) |
| Shared logic | [`sentiment_core.py`](./sentiment_core.py) |
| Web (FastAPI) | [`web/`](./web/) — port **3008** |

## Run the web demo

```powershell
Set-Location "c:\Users\emmak\Desktop\Content\CodingProjects\PortProjects\demos\user-review-sentiment\web"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 3008
```

Open [http://127.0.0.1:3008/](http://127.0.0.1:3008/). The first request loads or trains the model (cached afterward).

## Run the CLI

```powershell
Set-Location "c:\Users\emmak\Desktop\Content\CodingProjects\PortProjects\demos\user-review-sentiment"
pip install -r web/requirements.txt
python reference\run_cli.py
```
