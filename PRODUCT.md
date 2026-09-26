# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

The user uploads their own CSV or Excel dataset to inspect it.

## Product Purpose

Provide an easy way to run descriptive analysis and identify unusual rows in a dataset.

## Positioning

A local browser-based workflow combines data profiling with an unsupervised Isolation Forest review ranking.

## Operating Context

The user selects a local data file, reviews its profile and anomaly ranking, and downloads the generated reports.

## Capabilities and Constraints

The analysis runs on the user's computer. Supported formats are CSV and Excel. Anomaly scores are exploratory review rankings, not probabilities or determinations of fraud.

## Evidence on Hand

The existing Python analysis module profiles tabular data and creates an Excel report and anomaly-score CSV. A sample TradeStat export workbook is available for validation.

## Product Principles

- Make data selection and analysis available in one simple workflow.
- Show the dataset's shape, quality, and distributions before anomaly results.
- Keep uploaded data local to the user's running app.
- Describe model output as a review aid, not a ground-truth label.
