# Research: Sentiment model choice for Bluesky + future web/news investment sentiment

## Summary
For this project, **FinBERT is the best target model for finance/news sentiment**, but it is not the best first dependency for a small local MVP if low setup friction, tests, and offline fallback matter. The recommended MVP is a **hybrid architecture**: implement a deterministic local lexicon fallback first, preferably VADER, behind a `SentimentAnalyzer` interface, then add an optional FinBERT backend for higher-quality financial text once dependencies are available. CardiffNLP/Twitter RoBERTa is attractive for short social posts, but less finance-specific; LLM APIs are powerful for explanation and disambiguation but are not ideal as the default MVP path because of cost, keys, latency, and test brittleness.

## Findings
1. **FinBERT is finance-domain aligned and should be the main quality upgrade for investment sentiment** — FinBERT models such as `ProsusAI/finbert` and `yiyanghkust/finbert-tone` are trained/fine-tuned on financial text and classify text into positive, negative, and neutral labels, which maps directly to the project’s `sentiment_results` schema. This is especially suitable for earnings, analyst, macro, stock-news, and company-event language where generic sentiment models can misread phrases like “beat estimates”, “downgrade”, or “guidance cut”. [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert), [yiyanghkust/finbert-tone](https://huggingface.co/yiyanghkust/finbert-tone)

2. **FinBERT is less ideal as the only MVP model because it adds ML dependency and runtime friction** — Running FinBERT locally normally requires `transformers`, `torch`, model downloads, and more CPU/RAM/time than a lexicon approach. For a university project, this can complicate installation, CI tests, and reproducibility. A clean implementation should make FinBERT optional and keep tests independent from model downloads by mocking the analyzer or using a deterministic fallback. [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/index)

3. **CardiffNLP/Twitter RoBERTa is strong for short social posts but less finance-specific** — `cardiffnlp/twitter-roberta-base-sentiment-latest` is designed around social-media style text, so it can handle short informal posts better than many news-oriented models. However, it is not finance-specialized, so it may interpret investment jargon less accurately than FinBERT. It is a good alternative backend for Bluesky-only social sentiment, but not the best single model if the roadmap includes web/news finance content. [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)

4. **VADER is the best offline MVP fallback** — VADER is lightweight, deterministic, easy to test, and designed for social-media sentiment. It requires no heavy neural model and can run offline once the package is installed. It is not finance-aware, but it is good enough to validate the end-to-end architecture: raw documents → sentiment results → aggregates. It also supports stable unit tests because outputs are deterministic. [VADER paper / project](https://github.com/cjhutto/vaderSentiment)

5. **TextBlob is simple but weaker for this use case** — TextBlob is easy to use, but its generic polarity score is usually less appropriate for short finance/social text than VADER. It can be kept as a fallback-of-last-resort, but it should not be the primary recommendation for investment sentiment. [TextBlob documentation](https://textblob.readthedocs.io/)

6. **LLM/API sentiment can improve quality but should not be the default pipeline dependency** — LLMs can handle nuanced cases, sarcasm, entity disambiguation, and produce explanations, especially when a post mentions multiple companies. However, API-based sentiment introduces keys, cost, rate limits, privacy considerations, non-deterministic outputs, and slower tests. It is better as a later optional enrichment layer, for example: classify ambiguous high-impact posts or produce summaries after daily aggregation. [OpenAI API docs](https://platform.openai.com/docs/), [Anthropic Claude docs](https://docs.anthropic.com/)

7. **Best architecture is backend-swappable, not model-hardcoded** — The project should define a stable interface such as `SentimentAnalyzer.analyze(text) -> SentimentResult`, then provide multiple implementations: `VaderSentimentAnalyzer`, `FinbertSentimentAnalyzer`, and later `LLMSentimentAnalyzer`. This keeps the pipeline testable and allows the final app to choose quality vs. speed vs. cost per environment.

8. **Recommended scoring schema should be model-independent** — Store `sentiment_label`, `sentiment_score`, `confidence`, and `model_name`. Map labels consistently: positive → positive score, neutral → near 0, negative → negative score. This allows comparing VADER, FinBERT, and future LLM outputs without changing downstream aggregation tables.

## Recommendation

### Best MVP implementation
Implement **VADER first as the default local/offline analyzer**, then add **FinBERT as an optional advanced analyzer**.

Why this is the best MVP path:

- Lowest setup friction.
- Fast local execution.
- Easy deterministic tests.
- No model download required for basic pipeline validation.
- Allows the sentiment database schema and aggregation code to be implemented immediately.
- FinBERT can be plugged in later without changing ingestion/storage architecture.

### Best production-quality target
Use **FinBERT as the preferred analyzer for finance/news and investment-specific text**, especially for future Tavily/news/web data. For Bluesky-only short posts, compare FinBERT and CardiffNLP/Twitter RoBERTa empirically on a small labeled sample, but do not block MVP implementation on that comparison.

### Practical model priority
1. **VADER** — MVP default and offline fallback.
2. **FinBERT** — main finance-aware upgrade.
3. **CardiffNLP Twitter RoBERTa** — optional social-post comparison backend.
4. **LLM/API** — optional later enrichment for ambiguous/high-value posts.
5. **TextBlob** — not recommended except as a very simple educational baseline.

## Implementation shape

### Proposed files

```txt
nlp/
├── sentiment_analyzer.py      # interface + dataclass
├── vader_sentiment.py         # lightweight default backend
├── finbert_sentiment.py       # optional backend, lazy imports
└── text_cleaner.py            # normalize post text before inference

pipeline/
└── process_sentiment.py       # reads unprocessed raw_documents, writes sentiment_results
```

### Interface

```python
@dataclass(frozen=True)
class SentimentResult:
    label: str              # positive, negative, neutral
    score: float            # -1.0 to 1.0
    confidence: float       # 0.0 to 1.0
    model_name: str

class SentimentAnalyzer(Protocol):
    def analyze(self, text: str) -> SentimentResult:
        ...
```

### VADER mapping

```txt
compound >= 0.05   -> positive
compound <= -0.05  -> negative
otherwise          -> neutral
score              -> compound
confidence         -> abs(compound), with a small neutral confidence rule
model_name         -> vader
```

### FinBERT mapping

```txt
label              -> positive / negative / neutral
score              -> positive_prob - negative_prob
confidence         -> max class probability
model_name         -> ProsusAI/finbert or yiyanghkust/finbert-tone
```

### Dependency strategy

Use optional dependencies:

```txt
requirements.txt              # core pipeline, maybe vaderSentiment only
requirements-ml.txt           # transformers, torch
```

Or use lazy import in `finbert_sentiment.py`:

```python
try:
    from transformers import pipeline
except ImportError:
    raise RuntimeError("Install ML dependencies: pip install transformers torch")
```

### Test strategy

- Unit-test VADER/TextBlob-style analyzer with deterministic examples.
- Unit-test score/label mapping.
- Unit-test database writes with fake analyzer.
- Do **not** require FinBERT download in normal tests.
- Add an optional integration test skipped unless `RUN_ML_TESTS=1`.

Example:

```python
@unittest.skipUnless(os.getenv("RUN_ML_TESTS") == "1", "ML integration test disabled")
def test_finbert_smoke():
    ...
```

## Sources
- Kept: ProsusAI/finbert (https://huggingface.co/ProsusAI/finbert) — finance-domain BERT sentiment model directly relevant to investment/news sentiment.
- Kept: yiyanghkust/finbert-tone (https://huggingface.co/yiyanghkust/finbert-tone) — another finance-specific FinBERT variant useful for comparison.
- Kept: CardiffNLP Twitter RoBERTa sentiment latest (https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) — social-media sentiment baseline relevant to Bluesky-style posts.
- Kept: VADER GitHub/project (https://github.com/cjhutto/vaderSentiment) — lightweight social sentiment model suitable for offline MVP fallback.
- Kept: Hugging Face Transformers docs (https://huggingface.co/docs/transformers/index) — implementation reference for local transformer pipelines.
- Kept: TextBlob docs (https://textblob.readthedocs.io/) — simple baseline, but less appropriate than VADER.
- Kept: OpenAI API docs (https://platform.openai.com/docs/) and Anthropic docs (https://docs.anthropic.com/) — representative LLM/API option references.
- Dropped: Generic blog posts ranking sentiment models — excluded because they are often SEO-heavy and less reliable than primary model/docs pages.
- Dropped: Stock-market prediction articles using sentiment — excluded because the question is model selection/implementation, not trading alpha validation.

## Gaps
- No live web-search tool was available in this subagent environment, so source verification could not be refreshed in-session.
- The best final model for Bluesky specifically should be validated with a small hand-labeled sample of posts from this project, ideally 100-300 examples across companies.
- Sarcasm, spam, bots, and entity ambiguity remain open problems; these are better handled with later filtering/disambiguation layers, not the first sentiment model alone.

## Final actionable decision
Start implementation with:

```txt
Default analyzer: VADER
Optional advanced analyzer: FinBERT
Architecture: swappable SentimentAnalyzer interface
Tests: deterministic VADER/fake analyzer tests; skip FinBERT integration unless explicitly enabled
```

This gives the project a working sentiment pipeline immediately while preserving the path to finance-grade FinBERT quality.
