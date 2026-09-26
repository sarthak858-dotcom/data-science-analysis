# Data Science Analysis

A local, upload-first data profiling app for CSV and Excel files. It combines
descriptive statistics with an unsupervised Isolation Forest review ranking.

## Run locally

Requires Python 3.10 or newer.

```powershell
python -m pip install -r requirements.txt
python -m streamlit run data_analysis_app.py
```

On Windows, you can also launch `run_analysis_app.bat` after installing the
requirements. Open the local URL printed by Streamlit, normally
`http://127.0.0.1:8501`.

## Use the app

1. Add a CSV, XLSX, or XLSM file in the upload area.
2. Choose an Excel worksheet, if applicable, and set the header row.
3. Select **Analyze dataset**.
4. Review the overview, column distributions, correlations, and unusual rows.
5. Download the Excel analysis report or the full anomaly ranking as CSV.

The app binds to localhost and analyzes uploaded data in memory. It does not
send the data to an external analysis service. Avoid exposing the local app
port publicly.

## Analysis notes

The profile includes column types, missing values, summary statistics,
category counts, numeric correlations, duplicate rows, and data previews.
Isolation Forest ranks unusual rows from numeric columns and suitable
low-cardinality categorical columns. Identifier-like and high-cardinality
columns are excluded from model features. The review sensitivity controls the
model's approximate flagged fraction.

Anomaly scores and review flags are exploratory triage aids, not fraud
probabilities, diagnoses, or proof of data errors. Review flagged rows using
domain knowledge. Very small datasets may not support anomaly scoring.

## Supported files

- CSV
- Excel `.xlsx`
- Excel macro-enabled `.xlsm`

The default upload limit is 200 MB per file; adjust it in
`.streamlit/config.toml` if needed.

## Files

- `data_analysis_app.py` — Streamlit upload and results interface
- `universal_data_analysis.py` — reusable profiling, report, and ML functions
- `requirements.txt` — Python dependencies
- `run_analysis_app.bat` — Windows launcher
