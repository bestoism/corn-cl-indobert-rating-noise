# skripsi-corn-label-noise

Code for my undergraduate thesis: **Integrating Confident Learning and Ordinal Regression (CORN) on IndoBERT to handle label noise in Indonesian app review rating prediction.**

Work in progress. This is research code, not a library.

## What this is

App review ratings (1-5 stars) are often used as training labels, but they don't always match the review text. This project looks at whether that mismatch can be treated as label noise, and whether detecting it helps rating prediction.

The idea being tested:

- Use [Confident Learning](https://arxiv.org/abs/1911.00068) (via `cleanlab`) to flag likely mislabeled reviews.
- Build the proxy classifier for that step with the same ordinal architecture as the final model (IndoBERT + [CORN](https://arxiv.org/abs/2111.08851)), instead of a generic classifier.
- Compare two ways of handling flagged rows: hard pruning (drop everything flagged) and severity-aware pruning (drop only rows where the ordinal distance between the label and the proxy prediction is >= 2).
- Compare CORN against a plain cross-entropy baseline on raw, hard-pruned, and severity-pruned training data (6 scenarios x 3 seeds).

The design assumes the outcome could go either way. Null or negative results are treated as valid findings.

## Data

Reviews of three Indonesian apps (Tokopedia, Gojek, SeaBank) scraped from Google Play with a per-rating quota. The data is **not** included in this repo. `scripts/scrape_google_play.py` reproduces the collection, but results will differ depending on when it is run.

## Layout

```
scripts/
  scrape_google_play.py   # stratified scraping + inline language filter
  eda_dataset.py          # dataset EDA and locking (manifest + SHA-256)
src/
  config.py               # paths, hyperparameters, proxy registry
  preprocess.py           # minimal BERT-oriented text cleaning
  data_split.py           # 70/10/20 stratified + text-grouped split
  proxy.py                # proxy classifiers P1-P4 (OOF probabilities)
  clean.py                # Confident Learning, hard / severity-aware pruning
  human_validation.py     # blind human validation sample + agreement / kappa
  train.py, models.py     # CE and CORN training on IndoBERT
  significance.py         # Wilcoxon + Holm-Bonferroni, bootstrap CIs
  gold_test.py            # small human-verified test subset
  metrics.py, data.py
notebooks/
  PILOT_STUDY.ipynb       # Colab notebook that drives the pipeline
```

## Setup

Developed and run on Google Colab (T4 GPU) with Google Drive for storage.

```bash
git clone https://github.com/bestoism/skripsi-corn-label-noise
cd skripsi-corn-label-noise
pip install -r requirements.txt
```

Paths are set in `src/config.py`. On Colab it expects `/content/drive/MyDrive/SKRIPSI_CORN`; elsewhere it falls back to a local `local_data/` folder.

Set `SKRIPSI_DEBUG=1` for a tiny, fast run to check that the pipeline works. Results from debug mode are not meaningful.

## Notes

- Base model is `indobenchmark/indobert-base-p1` throughout.
- The notebook and some modules are mid-refactor and may not be fully in sync with each other. `src/config.py` is the source of truth.
- No results are published here yet.

## License

Code is released under the MIT License (see `LICENSE`). Scraped review data and third-party resources (e.g. the Colloquial Indonesian Lexicon) are not covered by this license and follow their own terms.