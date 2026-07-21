# Python Scratchpad

Personal workspace for data engineering experiments and small tasks.

## Structure

| Folder | Purpose |
|--------|---------|
| `dataverse_sync/` | Dataverse → Databricks ingestion & Power Automate config generation |
| `data_validation/` | Synapse vs Databricks source table comparisons |
| `exploration/` | Ad-hoc experiments, scripts, puzzles |
| `output/` | Generated artifacts — xlsx, csv, html (gitignored) |

## Setup

```bash
pip install pyodbc polars databricks-sql-connector requests openpyxl
```

Secrets live in `.env` (gitignored). To use in notebooks:

```python
import sys; sys.path.insert(0, "..")
import env_loader
import os

token = os.environ["DATABRICKS_DEV_TOKEN"]
```

## Key Environment Variables

| Variable | Description |
|----------|-------------|
| `DATAVERSE_CLIENT_ID` | Azure AD app registration |
| `DATAVERSE_CLIENT_SECRET` | Service principal secret |
| `DATABRICKS_DEV_TOKEN` | Databricks PAT (dev) |
| `DATABRICKS_TST_TOKEN` | Databricks PAT (tst) |
| `DATABRICKS_PRD_TOKEN` | Databricks PAT (prd) |
| `SYNAPSE_SERVER` | Synapse SQL endpoint |
