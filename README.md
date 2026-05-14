# Financial Analysis Pipeline

Pipeline modular para ingesta de texto financiero (redes sociales), detección de menciones por compañía y análisis de sentimiento con **FinBERT**.

## Objetivo

Construir una base de datos de sentimiento por empresa que sirva como capa de datos para análisis cuantitativo, comparación de modelos y futura API/dashboard.

## Arquitectura del repositorio

```txt
config/
  companies.json              # Watchlist de compañías/tickers/aliases

ingestion/
  bluesky_client.py           # Cliente async para app.bsky.feed.searchPosts

pipeline/
  ingest_bluesky.py           # CLI de ingesta + persistencia + mention detection
  process_sentiment.py        # CLI de filtrado de relevancia + scoring de sentimiento

nlp/
  mention_detector.py         # Detector rule-based (ticker/cashtag/nombre/alias)
  cleaner.py                  # Normalización de texto
  sentiment_analyzer.py       # FinBERT (default) + VADER fallback

storage/
  sqlite_store.py             # Esquema SQLite + operaciones CRUD/upsert

tests/
  test_pipeline_modules.py    # Tests unitarios e integración ligera

data/
  sentiment.db                # Base SQLite local
```

## Pipeline end-to-end

1. **Carga de compañías** (`config/companies.json`) en tabla `companies`.
2. **Ingesta Bluesky** (`pipeline.ingest_bluesky`):
   - consulta por compañía con tres patrones: `$TICKER`, `TICKER`, `Company stock`.
   - persiste posts en `raw_documents` con deduplicación por `(provider, external_id)`.
3. **Detección de menciones** (`nlp.mention_detector`):
   - detecta todas las empresas presentes en cada post.
   - persiste en `company_mentions` con deduplicación por `(company_id, raw_document_id, matched_text)`.
4. **Limpieza de texto** (`nlp.cleaner`):
   - elimina URLs y normaliza espacios.
5. **Filtrado de relevancia y sentiment scoring** (`pipeline.process_sentiment`):
   - por defecto usa **FinBERT** `ProsusAI/finbert`.
   - opcional: `--model vader`.
   - opcional: `--relevance-filter llm` usa `llama-server` para separar textos económicos de menciones irrelevantes.
   - persiste la relevancia en `document_relevance` y el sentimiento en `sentiment_results`.

## Modelo de sentimiento

- Default: `FinBertSentimentAnalyzer` (`model_name = finbert:ProsusAI/finbert`)
- Alternativo: `VaderSentimentAnalyzer` (`model_name = vader` o `vader-fallback`)

Mapeo de score en FinBERT:
- `positive` -> `+confidence`
- `negative` -> `-confidence`
- `neutral` -> `0.0`

## Esquema de datos (SQLite)

- `companies`
- `raw_documents`
- `company_mentions`
- `sentiment_results`
- `document_relevance`

La inicialización del esquema está en `storage/sqlite_store.py` (`SQLiteStore.init_schema`).

`document_relevance` guarda decisiones de relevancia por `(company_id, raw_document_id, classifier_name)`. Si `llama-server` marca un documento como irrelevante con `llm:llama-server`, futuras ejecuciones con `--relevance-filter llm` no vuelven a clasificarlo.

## Requisitos

- Python `>=3.12,<3.13`
- Dependencias definidas en `pyproject.toml`:
  - `torch`
  - `transformers`
  - `huggingface_hub`
  - `vadersentiment`

## Setup rápido

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Si usas `uv`:

```bash
uv sync
source .venv/bin/activate
```

## Comandos principales

### 1) Ingesta Bluesky

```bash
python3 -m pipeline.ingest_bluesky --limit 25
```

Opciones útiles:

```bash
python3 -m pipeline.ingest_bluesky --company TSLA --limit 50 --lang en --concurrency 3
```

### 2) Procesar sentimiento (FinBERT por defecto)

```bash
python3 -m pipeline.process_sentiment --limit 100 --model finbert
```

Con filtro LLM de relevancia económica:

```bash
python3 -m pipeline.process_sentiment --limit 1000000 --model finbert --relevance-filter llm --llama-concurrency 10
```

Modelo alternativo:

```bash
python3 -m pipeline.process_sentiment --limit 100 --model vader
```

### 3) Tests

```bash
python3 -m unittest discover -s tests -v
```

## Inspección rápida de la base

```bash
sqlite3 data/sentiment.db
```

Consultas útiles:

```sql
select model_name, count(*)
from sentiment_results
group by model_name;

select c.ticker, sr.sentiment_label, sr.sentiment_score, sr.confidence, sr.model_name
from sentiment_results sr
join companies c on c.id = sr.company_id
order by sr.id desc
limit 20;
```

## Etiquetado manual y comparación con FinBERT

Flujo recomendado:

1. Exportar CSV para etiquetado manual:

```bash
python3 scripts/export_manual_labels.py --db data/sentiment.db --output data/manual_labeling_finbert.csv
```

2. Etiquetar `manual_label` con valores exactos: `positive|neutral|negative`.

3. Evaluar métricas vs `finbert_label`:

```bash
python3 scripts/evaluate_manual_labels.py --csv data/manual_labeling_finbert.csv
```

Opcional: guardar reporte JSON para histórico:

```bash
python3 scripts/evaluate_manual_labels.py --csv data/manual_labeling_finbert.csv --output-json data/metrics/finbert_eval.json
```

Métricas reportadas:
- Accuracy
- Recall (macro, weighted y por clase)
- F1 score (macro, weighted y por clase)
- Matriz de confusión (manual -> finbert)

## Prueba final de evaluación

Archivos generados para el trabajo universitario:

- `docs/trabajo_final.md`: memoria redactada de la tarea de evaluación.
- `data/final_manual_review_100.csv`: 100 ejemplos con etiquetado manual de relevancia económica y sentimiento.
- `data/metrics/final_manual_review_100_eval.json`: métricas de FinBERT frente al etiquetado manual.

Reproducir evaluación final:

```bash
python3 scripts/evaluate_manual_labels.py --csv data/final_manual_review_100.csv --output-json data/metrics/final_manual_review_100_eval.json
```

Resultado de la muestra final: accuracy `0.6600`, recall macro `0.5879`, F1 macro `0.5823`.

## Estado actual

- Ingesta asíncrona desde Bluesky operativa.
- Detección de menciones multiempresa operativa.
- Filtro LLM de relevancia económica integrado con persistencia de irrelevantes.
- Sentimiento con FinBERT integrado en pipeline.
- Batería de tests unitaria/integración ligera en verde.
