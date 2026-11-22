# Order Data Cleaning + Baseline ML Pipeline

Two small, self-contained Python scripts I use to show how I handle data:
one turns a messy e-commerce export into a clean, documented file, and one
trains a model against a proper baseline so the numbers mean something.

Both run on CPU in seconds with no API keys and no downloads.

---

## What's inside

| File | What it does |
|---|---|
| `src/cleaning.py` | Cleans a messy orders CSV and writes a clean file plus a report of every decision |
| `src/train_model.py` | Trains and evaluates a classifier on a real public dataset, compared to a majority-class baseline |
| `tests/test_cleaning.py` | 14 unit tests covering each cleaning rule |
| `data/messy_orders.csv` | Demo dataset, deliberately dirty (see note below) |
| `reports/` | Real output from the last run, committed so you can check it without running anything |

---

## Results

### Cleaning — `data/messy_orders.csv`

17 rows in, **14 out**. Everything that changed is counted:

| Issue | Count | Action |
|---|---|---|
| Exact duplicate rows | 2 | removed |
| Text needing normalisation (case, stray spaces) | 5 | cleaned |
| Missing customer name | 1 | filled with `Unknown` |
| Missing unit price | 1 | row dropped |
| Missing quantity | 1 | imputed with the median |
| Negative quantity | 1 | converted to absolute value |
| Zero or negative price | 2 | kept, flagged, excluded from revenue |
| Future-dated order | 1 | flagged, not removed |
| Price outlier (IQR) | 1 | flagged, not removed |
| Dates that would not parse | 0 | — |

Full output: [`reports/cleaning_report.md`](reports/cleaning_report.md)

### Model — Wisconsin Diagnostic Breast Cancer (569 rows, 30 features)

Public dataset, bundled with scikit-learn. 75/25 stratified split.

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Baseline (majority class) | 0.6294 | 0.6294 | 1.0000 | 0.7725 | — |
| Logistic regression | **0.9860** | 0.9889 | 0.9889 | 0.9889 | 0.9977 |

5-fold cross-validation F1 on the training set: **0.9833 ± 0.0070**.

The baseline is the point of this table. A model that reports 98.6% accuracy
means nothing until you know the dumbest possible model already gets 62.9%.

---

## Try it

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# clean the demo data
python src\cleaning.py data\messy_orders.csv --out reports

# train and evaluate
python src\train_model.py

# run the tests
python -m unittest discover -s tests -v
```

To use the cleaner on your own file, point it at any CSV with the same columns:

```powershell
python src\cleaning.py path\to\your_export.csv --out out
```

---

## Design decisions

These are the parts worth arguing about.

**Nothing is deleted silently.** Every dropped, filled or flagged row is counted
in the report. If I remove a row, the client can see how many and why.

**Missing prices are dropped, missing quantities are imputed.** A missing price
cannot be reconstructed from the row, so inventing one would corrupt any revenue
total downstream. A missing quantity can be estimated from the other rows in the
same file, so it is imputed with the median and counted.

**Negative quantities become positive and stay in the file.** In a sales export
a negative quantity is usually a typo or a returned line. Deleting it loses the
record; keeping it as-is breaks any sum. Converting it and flagging it lets a
human decide.

**Zero and negative prices are flagged, not removed.** They usually mean a free
sample or a data-entry error. The row is kept for traceability but excluded from
`revenue`, so totals stay correct.

**Future dates are flagged, not removed.** They may be real pre-orders. Removing
them would quietly delete real business.

**Ambiguous dates are resolved day-first.** `05/01/2026` is 5 January under the
Egyptian/Gulf convention. This is an assumption, so the report lists which
format each row matched.

**Scaling lives inside the Pipeline.** In `train_model.py` the scaler is a step
in the same `Pipeline` as the model, so it is fitted on training data only. This
is what prevents data leakage, and it is the most common mistake in beginner ML
code.

---

## Tests

14 tests, one per rule, plus an end-to-end test on the sample file.

```powershell
python -m unittest discover -s tests -v
```

They cover: duplicate removal, text normalisation, missing customer, dropped
missing price, median imputation, negative quantity, zero price, revenue
calculation, all three date formats, unparseable dates, and future-date flagging.

---

## Limitations

Being straight about this matters more than the accuracy number.

- `data/messy_orders.csv` is **synthetic**. I built it to exercise every rule.
  It is not real customer data, and the counts above describe that demo file only.
- The breast-cancer dataset is small, clean and easy. 98.6% here says the method
  is correct — it does not say I can beat a specialist on a hard, messy,
  real-world problem.
- The cleaner assumes one row per order line and a fixed set of columns. A real
  export with nested order items would need a flattening step first.
- Outlier detection is plain IQR. On skewed prices (most stores) a log transform
  or a percentile cut is usually better.
- No schema validation yet: a column of the right name but the wrong type will
  fail at the cleaning step rather than at load.

---

## What I'd add next

1. Schema validation at load time (pandera or a small hand-written check) so bad
   input fails fast with a readable message.
2. A `--dry-run` flag that prints the report without writing the clean file.
3. Swap IQR for a configurable strategy, and log which strategy was used.
4. A tiny CLI summary at the end: rows fixed, rows flagged, rows dropped, as a
   single line a human can read in a log.

---

*Built as a working sample. If something in here is wrong or unclear, I would
rather hear it than have it stay wrong.*
