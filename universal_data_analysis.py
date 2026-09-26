"""Descriptive profiling and unsupervised anomaly ranking for tabular data.

Examples:
    python universal_data_analysis.py "sales.xlsx"
    python universal_data_analysis.py "sales.csv" --output-dir "reports"
    python universal_data_analysis.py "sales.xlsx" --sheet "Data" --header-row 2

Dependencies:
    python -m pip install pandas numpy scikit-learn openpyxl xlsxwriter
"""

from __future__ import annotations

import argparse
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


MAX_CATEGORIES = 20
MAX_TRAIN_ROWS = 30_000
MAX_REPORT_ROWS = 1_000
RANDOM_STATE = 42
ID_NAME = re.compile(
    r"(^|[_\W])(id|uuid|guid|index|serial|s\.?\s*no\.?|row[_ ]?number)([_\W]|$)",
    re.I,
)


def read_table(
    path: Path, sheet: str | int | None, header_row: int
) -> pd.DataFrame:
    """Load CSV or Excel data, preserving source columns."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, sep=None, engine="python", header=header_row)
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(
            path,
            sheet_name=sheet if sheet is not None else 0,
            header=header_row,
        )
    raise ValueError("Supported input formats are .csv, .xlsx, .xlsm, and .xls.")


def infer_column_types(
    df: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    """Infer numeric and date-like text columns; retain other columns as categorical."""
    numeric: list[str] = []
    dates: list[str] = []
    categorical: list[str] = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_bool_dtype(series):
            categorical.append(column)
        elif pd.api.types.is_numeric_dtype(series):
            numeric.append(column)
        elif pd.api.types.is_datetime64_any_dtype(series):
            dates.append(column)
        else:
            non_null = series.dropna()
            if non_null.empty:
                categorical.append(column)
                continue
            text = non_null.astype(str).str.strip()
            numeric_text = pd.to_numeric(
                text.str.replace(",", "", regex=False)
                .str.replace(r"^[^\d+\-.]*", "", regex=True)
                .str.replace(r"[^\d.]+$", "", regex=True),
                errors="coerce",
            )
            numeric_rate = numeric_text.notna().mean()
            date_rate = 0.0
            if numeric_rate < 0.8 and (
                re.search(r"date|time|timestamp", str(column), re.I)
                or text.str.contains(r"[-/:]", regex=True).mean() > 0.5
            ):
                parsed = pd.to_datetime(text, errors="coerce", format="mixed")
                date_rate = parsed.notna().mean()
            if numeric_rate >= 0.9:
                df[column] = pd.to_numeric(
                    series.astype("string")
                    .str.replace(",", "", regex=False)
                    .str.replace(r"^[^\d+\-.]*", "", regex=True)
                    .str.replace(r"[^\d.]+$", "", regex=True),
                    errors="coerce",
                )
                numeric.append(column)
            elif date_rate >= 0.8:
                df[column] = pd.to_datetime(series, errors="coerce", format="mixed")
                dates.append(column)
            else:
                categorical.append(column)
    return numeric, categorical, dates


def make_numeric_features(
    df: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    for column in columns:
        if ID_NAME.search(str(column)):
            continue
        values = pd.to_numeric(df[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if values.notna().any() and values.nunique(dropna=True) > 1:
            features[f"num::{column}"] = values.fillna(values.median())
    return features


def prepare_ml_features(
    df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Build a bounded, numeric matrix for Isolation Forest."""
    numeric_features = make_numeric_features(df, numeric_columns)
    included_categories: list[str] = []
    categorical_frames: list[pd.DataFrame] = []

    for column in categorical_columns:
        values = df[column].astype("string").fillna("__MISSING__").str.strip()
        distinct = values.nunique(dropna=False)
        unique_ratio = distinct / max(len(values), 1)
        if distinct < 2:
            continue
        if distinct > 100 or unique_ratio > 0.5 or ID_NAME.search(str(column)):
            continue

        top_values = values.value_counts().head(MAX_CATEGORIES).index
        reduced = values.where(values.isin(top_values), "__OTHER__")
        encoded = pd.get_dummies(reduced, prefix=str(column), dtype=float)
        categorical_frames.append(encoded)
        included_categories.append(column)

    pieces = [f for f in (numeric_features, *categorical_frames) if not f.empty]
    if not pieces:
        return pd.DataFrame(index=df.index), included_categories
    features = pd.concat(pieces, axis=1)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features, included_categories


def numeric_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        valid = values.dropna()
        if valid.empty:
            continue
        rows.append(
            {
                "column": column,
                "count": int(valid.count()),
                "missing": int(values.isna().sum()),
                "missing_percent": float(values.isna().mean() * 100),
                "mean": float(valid.mean()),
                "std_dev": float(valid.std()) if len(valid) > 1 else 0.0,
                "min": float(valid.min()),
                "p25": float(valid.quantile(0.25)),
                "median": float(valid.median()),
                "p75": float(valid.quantile(0.75)),
                "max": float(valid.max()),
                "unique_values": int(valid.nunique()),
            }
        )
    return pd.DataFrame(rows)


def categorical_summary(
    df: pd.DataFrame, columns: list[str], top_n: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in columns:
        values = df[column].astype("string").fillna("(missing)")
        counts = values.value_counts(dropna=False)
        for value, count in counts.head(top_n).items():
            rows.append(
                {
                    "column": column,
                    "value": str(value),
                    "count": int(count),
                    "percent": float(count / len(df) * 100) if len(df) else 0.0,
                    "distinct_values_in_column": int(counts.size),
                }
            )
    return pd.DataFrame(rows)


def column_summary(
    df: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    dates: list[str],
    model_categories: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in df.columns:
        values = df[column]
        non_null = values.dropna()
        kind = (
            "numeric"
            if column in numeric
            else "date/time"
            if column in dates
            else "categorical"
        )
        unique = int(values.nunique(dropna=True))
        note = ""
        if kind == "categorical" and column not in model_categories:
            if ID_NAME.search(str(column)):
                note = "Identifier-like column; excluded from model"
            elif unique > 100 or unique / max(len(df), 1) > 0.5:
                note = "High-cardinality; excluded from model"
            elif unique < 2:
                note = "Constant column; excluded from model"
        elif kind == "numeric" and ID_NAME.search(str(column)):
            note = "Identifier-like column; excluded from model"
        elif kind == "numeric" and unique < 2:
            note = "Constant column; excluded from model"
        rows.append(
            {
                "column": column,
                "type": kind,
                "non_missing": int(values.notna().sum()),
                "missing": int(values.isna().sum()),
                "missing_percent": float(values.isna().mean() * 100)
                if len(df)
                else 0.0,
                "unique_values": unique,
                "example_values": ", ".join(map(str, non_null.head(3).tolist())),
                "model_note": note,
            }
        )
    return pd.DataFrame(rows)


def add_anomaly_scores(
    features: pd.DataFrame, contamination: float
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    if features.empty or len(features) < 20:
        return (
            None,
            None,
            "Skipped: at least 20 rows and one usable feature are required.",
        )

    # Robust scaling prevents large-unit numeric fields from dominating the feature space.
    scaler = RobustScaler()
    sample_size = min(len(features), MAX_TRAIN_ROWS)
    if sample_size < len(features):
        sample = features.sample(n=sample_size, random_state=RANDOM_STATE)
    else:
        sample = features
    scaler.fit(sample)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(scaler.transform(sample))
    # Larger values mean more unusual; scores are rankings, not fraud probabilities.
    transformed = scaler.transform(features)
    scores = -model.decision_function(transformed)
    flags = model.predict(transformed) == -1
    method = (
        f"Isolation Forest; trained on {sample_size:,} rows; "
        f"{features.shape[1]:,} model features."
    )
    return scores, flags, method


def write_report(
    output_path: Path | BytesIO,
    df: pd.DataFrame,
    columns: pd.DataFrame,
    numeric: pd.DataFrame,
    categorical: pd.DataFrame,
    correlations: pd.DataFrame,
    anomalies: pd.DataFrame,
    method: str,
) -> None:
    overview = pd.DataFrame(
        [
            ("Rows", len(df)),
            ("Columns", len(df.columns)),
            ("Duplicate rows", int(df.duplicated().sum())),
            ("Cells missing", int(df.isna().sum().sum())),
            ("Missing cells (%)", float(df.isna().to_numpy().mean() * 100)),
            ("Anomaly scoring", method),
        ],
        columns=["Metric", "Result"],
    )
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        overview.to_excel(writer, sheet_name="Overview", index=False)
        columns.to_excel(writer, sheet_name="Columns", index=False)
        numeric.to_excel(writer, sheet_name="Numeric Stats", index=False)
        categorical.to_excel(writer, sheet_name="Category Counts", index=False)
        correlations.to_excel(writer, sheet_name="Correlations", index=True)
        anomalies.head(MAX_REPORT_ROWS).to_excel(
            writer, sheet_name="Top Anomalies", index=False
        )

        workbook = writer.book
        title = workbook.add_format(
            {"bold": True, "font_color": "#FFFFFF", "bg_color": "#1E4468"}
        )
        for sheet_name, frame in [
            ("Overview", overview),
            ("Columns", columns),
            ("Numeric Stats", numeric),
            ("Category Counts", categorical),
            ("Correlations", correlations.reset_index()),
            ("Top Anomalies", anomalies.head(MAX_REPORT_ROWS)),
        ]:
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes(1, 0)
            worksheet.set_row(0, 28, title)
            worksheet.autofilter(
                0,
                0,
                max(len(frame), 1),
                max(len(frame.columns) - 1, 0),
            )
            worksheet.set_column(0, max(len(frame.columns) - 1, 0), 20)
            if sheet_name in {"Columns", "Category Counts", "Top Anomalies"}:
                worksheet.set_column(0, 0, 28)
                worksheet.set_column(1, max(len(frame.columns) - 1, 1), 24)
        writer.sheets["Overview"].set_column("A:A", 24)
        writer.sheets["Overview"].set_column("B:B", 72)


def analyze(args: argparse.Namespace) -> tuple[Path, Path | None]:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    sheet: str | int | None = args.sheet
    if sheet is not None and str(sheet).isdigit():
        sheet = int(sheet)

    df = read_table(input_path, sheet, args.header_row)
    if df.empty:
        raise ValueError("The selected file or sheet contains no data rows.")
    df.columns = [str(c).strip() or f"unnamed_{i}" for i, c in enumerate(df.columns)]
    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names must be resolved first: {duplicates}")

    numeric_columns, categorical_columns, date_columns = infer_column_types(df)
    model_features, model_categories = prepare_ml_features(
        df, numeric_columns, categorical_columns
    )
    scores, review_flags, method = add_anomaly_scores(
        model_features, args.contamination
    )

    columns = column_summary(
        df, numeric_columns, categorical_columns, date_columns, model_categories
    )
    numeric = numeric_summary(df, numeric_columns)
    categorical = categorical_summary(df, categorical_columns, args.top_categories)
    corr_columns = [
        c for c in numeric_columns if df[c].nunique(dropna=True) > 1
    ][:50]
    correlations = df[corr_columns].corr(numeric_only=True) if corr_columns else pd.DataFrame()

    anomalies = pd.DataFrame()
    if scores is not None and review_flags is not None:
        anomalies = df.copy()
        first_data_row = args.header_row + 2
        anomalies.insert(
            0,
            "source_row",
            np.arange(first_data_row, first_data_row + len(df)),
        )
        anomalies["anomaly_score"] = scores
        anomalies["anomaly_rank"] = (
            anomalies["anomaly_score"].rank(method="first", ascending=False).astype(int)
        )
        anomalies["review_flag"] = review_flags
        anomalies = anomalies.sort_values(
            "anomaly_score", ascending=False, kind="stable"
        )

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base = input_path.stem
    report_path = out_dir / f"{base}_analysis_report.xlsx"
    write_report(
        report_path,
        df,
        columns,
        numeric,
        categorical,
        correlations,
        anomalies,
        method,
    )
    anomaly_path: Path | None = None
    if not anomalies.empty:
        anomaly_path = out_dir / f"{base}_anomaly_scores.csv"
        anomalies.to_csv(anomaly_path, index=False, encoding="utf-8-sig")
    return report_path, anomaly_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile tabular CSV/Excel data and rank unusual rows with Isolation Forest."
    )
    parser.add_argument("input", help="Input CSV, XLSX, XLSM, or XLS file")
    parser.add_argument(
        "--sheet",
        help="Excel sheet name or zero-based sheet number (default: first sheet)",
    )
    parser.add_argument(
        "--header-row",
        type=int,
        default=0,
        help="Zero-based row containing column names (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        help="Folder for reports (default: next to the input file)",
    )
    parser.add_argument(
        "--top-categories",
        type=int,
        default=MAX_CATEGORIES,
        help=f"Top category values per column to report (default: {MAX_CATEGORIES})",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Expected anomaly fraction for review_flag, from 0.001 to 0.5 (default: 0.05)",
    )
    args = parser.parse_args()
    if args.top_categories < 1:
        parser.error("--top-categories must be at least 1")
    if args.header_row < 0:
        parser.error("--header-row must be zero or greater")
    if not 0.001 <= args.contamination <= 0.5:
        parser.error("--contamination must be between 0.001 and 0.5")
    try:
        report_path, anomaly_path = analyze(args)
    except (OSError, ValueError, ImportError) as exc:
        print(f"Analysis failed: {exc}", file=sys.stderr)
        return 1
    print(f"Analysis report: {report_path}")
    if anomaly_path:
        print(f"Full anomaly ranking: {anomaly_path}")
    else:
        print("Anomaly scoring was skipped; see Overview in the report for details.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
