"""Local upload-first descriptive analysis app.

THESIS: Make a dataset inspectable immediately; refuse the blank upload-only dashboard.
OWN-WORLD: A quiet analyst's field notebook: paper-white workspace, ink-blue structure,
and a single measured teal for selected evidence and model review.
STORY: Select a local file, inspect its shape and data quality, then review unusual rows.
FIRST VIEWPORT: The dataset title and file picker lead; analysis settings sit directly
below, with the run action beside them and profile results following in the same column.
FORM: A field notebook workbench, third grounded direction; compact upload-to-review
workflow, with familiar Streamlit controls and responsive stacked sections.
"""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from universal_data_analysis import (
    add_anomaly_scores,
    categorical_summary,
    column_summary,
    infer_column_types,
    numeric_summary,
    prepare_ml_features,
    write_report,
)


MAX_FILE_SIZE_MB = 200
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xlsm"}
DEFAULT_CSS = """
<style>
  :root { color-scheme: light; }
  .stApp { background: #f5f7f8; color: #142b3c; }
  [data-testid="stHeader"] { background: rgba(245,247,248,.94); }
  [data-testid="stSidebar"] { background: #eaf0f2; }
  .block-container { max-width: 1240px; padding-top: 2.25rem; padding-bottom: 4rem; }
  h1, h2, h3 { color: #142b3c; letter-spacing: -.025em; }
  div[data-testid="stMetric"] {
    background: #fff; border: 1px solid #d9e2e7; border-radius: 8px;
    padding: 14px 16px;
  }
  div[data-testid="stMetricLabel"] { color: #526879; }
  .stButton > button[kind="primary"] {
    background: #176e72; border-color: #176e72; color: #fff;
  }
  .stButton > button[kind="primary"]:hover {
    background: #11585c; border-color: #11585c; color: #fff;
  }
  [data-testid="stFileUploaderDropzone"] {
    background: #fff; border: 1px dashed #91a9b5; border-radius: 8px;
  }
  @media (max-width: 700px) {
    .block-container { padding: 1.1rem 1rem 2rem; }
  }
</style>
"""


@dataclass
class Analysis:
    frame: pd.DataFrame
    columns: pd.DataFrame
    numeric: pd.DataFrame
    categorical: pd.DataFrame
    correlations: pd.DataFrame
    anomalies: pd.DataFrame
    scored_count: int
    flagged_count: int
    method: str
    report_bytes: bytes
    anomaly_csv: bytes | None


def parse_uploaded_file(
    raw: bytes, filename: str, sheet: str | int, header_row: int
) -> pd.DataFrame:
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    buffer = io.BytesIO(raw)
    if suffix == ".csv":
        return pd.read_csv(buffer, sep=None, engine="python", header=header_row)
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(buffer, sheet_name=sheet, header=header_row)
    raise ValueError("Choose a CSV, XLSX, or XLSM file.")


def build_analysis(
    raw: bytes,
    filename: str,
    sheet: str | int,
    header_row: int,
    top_categories: int,
    contamination: float,
) -> Analysis:
    frame = parse_uploaded_file(raw, filename, sheet, header_row)
    if frame.empty:
        raise ValueError("The selected file or worksheet has no data rows.")

    frame.columns = [
        str(column).strip() or f"unnamed_{index}"
        for index, column in enumerate(frame.columns)
    ]
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Column names must be unique. Duplicates: {duplicates}")

    numeric_columns, categorical_columns, date_columns = infer_column_types(frame)
    features, model_categories = prepare_ml_features(
        frame, numeric_columns, categorical_columns
    )
    scores, review_flags, method = add_anomaly_scores(features, contamination)

    columns = column_summary(
        frame,
        numeric_columns,
        categorical_columns,
        date_columns,
        model_categories,
    )
    numeric = numeric_summary(frame, numeric_columns)
    categorical = categorical_summary(frame, categorical_columns, top_categories)
    correlation_columns = [
        column for column in numeric_columns if frame[column].nunique(dropna=True) > 1
    ][:50]
    correlations = (
        frame[correlation_columns].corr(numeric_only=True)
        if correlation_columns
        else pd.DataFrame()
    )

    anomalies = pd.DataFrame()
    anomaly_csv: bytes | None = None
    if scores is not None and review_flags is not None:
        anomalies = frame.copy()
        source_row_start = header_row + 2
        anomalies.insert(
            0,
            "source_row",
            np.arange(source_row_start, source_row_start + len(frame)),
        )
        anomalies["anomaly_score"] = scores
        anomalies["anomaly_rank"] = (
            anomalies["anomaly_score"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        anomalies["review_flag"] = review_flags
        anomalies.sort_values(
            "anomaly_score", ascending=False, kind="stable", inplace=True
        )
        csv_buffer = io.StringIO()
        anomalies.to_csv(csv_buffer, index=False)
        anomaly_csv = csv_buffer.getvalue().encode("utf-8-sig")

    report = io.BytesIO()
    write_report(
        report,
        frame,
        columns,
        numeric,
        categorical,
        correlations,
        anomalies,
        method,
    )
    return Analysis(
        frame=frame,
        columns=columns,
        numeric=numeric,
        categorical=categorical,
        correlations=correlations,
        anomalies=anomalies,
        scored_count=len(anomalies),
        flagged_count=int(anomalies["review_flag"].sum())
        if not anomalies.empty
        else 0,
        method=method,
        report_bytes=report.getvalue(),
        anomaly_csv=anomaly_csv,
    )


@st.cache_data(show_spinner=False, max_entries=8)
def cached_analysis(
    raw: bytes,
    filename: str,
    sheet: str | int,
    header_row: int,
    top_categories: int,
    contamination: float,
) -> Analysis:
    return build_analysis(
        raw, filename, sheet, header_row, top_categories, contamination
    )


def workbook_sheets(raw: bytes) -> list[str]:
    try:
        return pd.ExcelFile(io.BytesIO(raw)).sheet_names
    except (ValueError, OSError, zipfile.BadZipFile, ImportError) as exc:
        raise ValueError(f"Could not read this Excel workbook: {exc}") from exc


def format_size(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    return f"{size / 1024:.0f} KB"


def show_summary(analysis: Analysis) -> None:
    frame = analysis.frame
    missing_percent = (
        float(frame.isna().to_numpy().mean() * 100) if frame.size else 0.0
    )
    st.subheader("At a glance")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Rows", f"{len(frame):,}")
    metric_columns[1].metric("Columns", f"{len(frame.columns):,}")
    metric_columns[2].metric("Missing cells", f"{missing_percent:.2f}%")
    metric_columns[3].metric(
        "Review flags",
        f"{analysis.flagged_count:,}" if analysis.scored_count else "Not scored",
    )

    quality_left, quality_right = st.columns([1.1, 1])
    with quality_left:
        st.markdown("#### Missing values")
        missing = (
            frame.isna()
            .sum()
            .rename("Missing rows")
            .to_frame()
            .query("`Missing rows` > 0")
            .sort_values("Missing rows", ascending=False)
            .head(12)
        )
        if missing.empty:
            st.success("No missing values found.")
        else:
            st.bar_chart(missing, horizontal=True, height=310)
    with quality_right:
        st.markdown("#### Analysis coverage")
        st.write(f"**{len(analysis.numeric):,}** numeric columns summarized")
        st.write(
            f"**{analysis.columns['type'].eq('categorical').sum():,}** "
            "categorical columns profiled"
        )
        st.write(
            f"**{len(analysis.correlations):,}** numeric columns included "
            "in the correlation table (up to 50)"
        )
        duplicates = int(frame.duplicated().sum())
        st.write(f"**{duplicates:,}** exact duplicate rows")
        st.caption(analysis.method)

    st.markdown("#### Data preview")
    st.dataframe(frame.head(12), use_container_width=True, hide_index=True)


def show_column_analysis(analysis: Analysis) -> None:
    st.subheader("Columns")
    st.dataframe(analysis.columns, use_container_width=True, hide_index=True)

    if not analysis.numeric.empty:
        st.markdown("#### Numeric distributions")
        numeric_columns = analysis.numeric["column"].tolist()
        selected = st.selectbox(
            "Choose a numeric column",
            numeric_columns,
            key="numeric_distribution_column",
        )
        values = pd.to_numeric(analysis.frame[selected], errors="coerce").dropna()
        if len(values):
            bin_count = min(24, max(5, int(np.sqrt(len(values)))))
            counts, edges = np.histogram(values, bins=bin_count)
            labels = [
                f"{edges[index]:,.3g}–{edges[index + 1]:,.3g}"
                for index in range(len(counts))
            ]
            distribution = pd.DataFrame({"Range": labels, "Rows": counts}).set_index(
                "Range"
            )
            st.bar_chart(distribution, height=300)
        st.dataframe(analysis.numeric, use_container_width=True, hide_index=True)
    else:
        st.info("No numeric columns were detected.")

    if not analysis.categorical.empty:
        st.markdown("#### Category counts")
        choices = analysis.categorical["column"].drop_duplicates().tolist()
        selected_category = st.selectbox(
            "Choose a category",
            choices,
            key="category_distribution_column",
        )
        counts = analysis.categorical.loc[
            analysis.categorical["column"].eq(selected_category),
            ["value", "count"],
        ].set_index("value")
        st.bar_chart(counts.head(15), horizontal=True, height=350)
        with st.expander("View reported category values"):
            st.dataframe(
                analysis.categorical.loc[
                    analysis.categorical["column"].eq(selected_category)
                ],
                use_container_width=True,
                hide_index=True,
            )

    if not analysis.correlations.empty:
        st.markdown("#### Numeric correlations")
        st.caption(
            "Pearson correlation ranges from -1 to 1. Correlation alone does not "
            "establish cause and effect."
        )
        st.dataframe(
            analysis.correlations.style.background_gradient(
                cmap="RdBu_r", vmin=-1, vmax=1
            ).format("{:.2f}", na_rep="—"),
            use_container_width=True,
        )


def show_anomalies(analysis: Analysis) -> None:
    st.subheader("Unusual rows")
    if analysis.anomalies.empty:
        st.info(
            "Isolation Forest needs at least 20 rows and one usable numeric or "
            "low-cardinality categorical feature."
        )
        return

    st.warning(
        "Anomaly scores rank unusual rows; they are not fraud probabilities or "
        "confirmed errors. Review flagged rows in context."
    )
    flagged_only = st.checkbox("Show review-flagged rows only", value=True)
    display_rows = analysis.anomalies
    if flagged_only:
        display_rows = display_rows.loc[display_rows["review_flag"]]
    st.caption(
        f"Showing {min(len(display_rows), 500):,} of {len(display_rows):,} "
        f"{'flagged ' if flagged_only else 'ranked '}rows."
    )
    st.dataframe(
        display_rows.head(500),
        use_container_width=True,
        hide_index=True,
        column_config={
            "anomaly_score": st.column_config.NumberColumn(format="%.5f"),
            "anomaly_rank": st.column_config.NumberColumn(format="%d"),
            "review_flag": st.column_config.CheckboxColumn(),
        },
    )
    if analysis.anomaly_csv is not None:
        st.download_button(
            "Download full anomaly ranking (CSV)",
            data=analysis.anomaly_csv,
            file_name="anomaly_scores.csv",
            mime="text/csv",
            key="download_anomaly_csv",
        )


def main() -> None:
    st.set_page_config(
        page_title="Data Fieldnotes",
        page_icon="▦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(DEFAULT_CSS, unsafe_allow_html=True)
    st.title("Read your data.")
    st.markdown(
        "Add a CSV or Excel file to see its shape, quality, distributions, "
        "and a machine-learning review of unusual rows."
    )
    st.caption(
        "Files are analyzed by this local app on your computer and are not "
        "uploaded to an analysis service."
    )

    uploaded_files = st.file_uploader(
        "Choose data files",
        type=["csv", "xlsx", "xlsm"],
        accept_multiple_files=True,
        help=f"CSV, XLSX, and XLSM · up to {MAX_FILE_SIZE_MB} MB per file",
    )
    if not uploaded_files:
        st.info(
            "Choose one or more files above to begin. Your files stay in this "
            "local app; select a worksheet and header row, then run the analysis."
        )
        return

    filenames = [item.name for item in uploaded_files]
    duplicate_names = {
        name for name, count in Counter(filenames).items() if count > 1
    }
    file_options = [
        f"{item.name} · file {index + 1}"
        if item.name in duplicate_names
        else item.name
        for index, item in enumerate(uploaded_files)
    ]
    selected_option = st.selectbox(
        "Dataset to analyze",
        file_options,
        key="selected_dataset_name",
    )
    uploaded = uploaded_files[file_options.index(selected_option)]
    raw = uploaded.getvalue()
    if len(raw) > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"{uploaded.name} exceeds the {MAX_FILE_SIZE_MB} MB file limit.")
        return
    if not raw:
        st.error(f"{uploaded.name} is empty.")
        return

    suffix = "." + uploaded.name.rsplit(".", 1)[-1].lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        st.error("Choose a CSV, XLSX, or XLSM file.")
        return
    sheets: list[str] = []
    if suffix in {".xlsx", ".xlsm"}:
        try:
            sheets = workbook_sheets(raw)
        except ValueError as exc:
            st.error(str(exc))
            return
        if not sheets:
            st.error("No worksheets were found in this workbook.")
            return

    with st.form("analysis_settings"):
        settings_left, settings_middle, settings_right = st.columns([1.1, 1, 1])
        with settings_left:
            selected_sheet: str | int = (
                st.selectbox("Worksheet", sheets) if sheets else 0
            )
        with settings_middle:
            header_row = (
                st.number_input(
                    "Header row",
                    min_value=1,
                    max_value=1000,
                    value=1,
                    help="The row containing the column names.",
                )
                - 1
            )
        with settings_right:
            contamination = st.slider(
                "Review sensitivity",
                min_value=1,
                max_value=20,
                value=5,
                format="%d%%",
                help=(
                    "Approximate fraction of rows flagged by Isolation Forest. "
                    "Flags are review prompts, not confirmed anomalies."
                ),
            )
        top_categories = st.slider(
            "Category values to summarize",
            min_value=5,
            max_value=30,
            value=20,
        )
        run_analysis = st.form_submit_button(
            "Analyze dataset", type="primary", use_container_width=True
        )

    config_fingerprint = (
        f"|{selected_option}|{selected_sheet}|{header_row}|"
        f"{top_categories}|{contamination}"
    ).encode("utf-8")
    fingerprint = hashlib.sha256(raw + config_fingerprint).hexdigest()
    state_key = "data_fieldnotes_result"
    if run_analysis:
        with st.spinner("Reading columns and building the analysis…"):
            try:
                result = cached_analysis(
                    raw,
                    uploaded.name,
                    selected_sheet,
                    int(header_row),
                    top_categories,
                    contamination / 100,
                )
            except (
                OSError,
                ValueError,
                UnicodeDecodeError,
                pd.errors.ParserError,
                zipfile.BadZipFile,
                ImportError,
            ) as exc:
                st.error(f"Could not analyze this file: {exc}")
                return
            st.session_state[state_key] = (fingerprint, result)

    saved = st.session_state.get(state_key)
    if not saved or saved[0] != fingerprint:
        if not run_analysis:
            file_detail = (
                f"{len(sheets)} worksheet(s)" if sheets else "CSV file"
            )
            st.caption(
                f"{uploaded.name} · {format_size(len(raw))} · {file_detail}"
            )
        return
    analysis: Analysis = saved[1]

    heading_left, heading_right = st.columns([3, 1])
    with heading_left:
        st.subheader(uploaded.name)
    with heading_right:
        st.download_button(
            "Download analysis report (Excel)",
            data=analysis.report_bytes,
            file_name=f"{re.sub(r'[^A-Za-z0-9._-]+', '_', uploaded.name.rsplit('.', 1)[0])}_analysis_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="download_analysis_report",
        )

    overview, columns, anomalies, preview = st.tabs(
        ["Overview", "Columns & distributions", "Anomaly review", "Data preview"]
    )
    with overview:
        show_summary(analysis)
    with columns:
        show_column_analysis(analysis)
    with anomalies:
        show_anomalies(analysis)
    with preview:
        st.subheader("Source rows")
        st.caption(f"Showing the first 500 of {len(analysis.frame):,} rows.")
        st.dataframe(
            analysis.frame.head(500),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
