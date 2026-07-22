"""Synapse vs Databricks data validation utilities.

Compares schema, row counts, and summary statistics via SQL queries.
No row-level data is downloaded.
"""

import polars as pl


DATABRICKS_TO_SYNAPSE_TYPE_MAP = {
    "TINYINT": "TINYINT", "SMALLINT": "SMALLINT", "INT": "INT", "INTEGER": "INT",
    "BIGINT": "BIGINT", "FLOAT": "REAL", "REAL": "REAL", "DOUBLE": "FLOAT",
    "DECIMAL": "DECIMAL", "NUMERIC": "NUMERIC", "STRING": "VARCHAR",
    "VARCHAR": "VARCHAR", "CHAR": "CHAR", "BINARY": "VARBINARY",
    "DATE": "DATE", "TIMESTAMP": "DATETIME2", "TIMESTAMP_NTZ": "DATETIME2",
    "TIMESTAMP_LTZ": "DATETIMEOFFSET", "BOOLEAN": "BIT",
}

NUMERIC_TYPES = {"INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT", "FLOAT",
                 "REAL", "DOUBLE", "DECIMAL", "NUMERIC", "MONEY", "SMALLMONEY"}
DATE_TYPES = {"DATE", "DATETIME", "DATETIME2", "DATETIMEOFFSET", "SMALLDATETIME",
              "TIMESTAMP", "TIMESTAMP_NTZ", "TIMESTAMP_LTZ"}
STRING_TYPES = {"VARCHAR", "NVARCHAR", "CHAR", "NCHAR", "TEXT", "NTEXT", "STRING"}


def base_type(dtype: str) -> str:
    """Extract base type name, stripping precision/scale."""
    return str(dtype).split("(")[0].strip().upper() if dtype is not None else None


def _build_synapse_stats_sql(cols: list[dict], schema: str, table: str) -> str:
    """Build a Synapse SQL query that computes per-column summary stats."""
    parts = []
    for col in cols:
        name, btype = col["name"], col["base_type"]
        q = f"[{name}]"
        parts.append(f"SUM(CASE WHEN {q} IS NULL THEN 1 ELSE 0 END) AS [{name}__nulls]")
        parts.append(f"COUNT(DISTINCT {q}) AS [{name}__distinct]")
        if btype in NUMERIC_TYPES:
            parts.append(f"MIN(CAST({q} AS FLOAT)) AS [{name}__min]")
            parts.append(f"MAX(CAST({q} AS FLOAT)) AS [{name}__max]")
            parts.append(f"AVG(CAST({q} AS FLOAT)) AS [{name}__avg]")
        elif btype in DATE_TYPES:
            parts.append(f"MIN({q}) AS [{name}__min]")
            parts.append(f"MAX({q}) AS [{name}__max]")
        elif btype in STRING_TYPES:
            parts.append(f"MIN(LEN({q})) AS [{name}__len_min]")
            parts.append(f"MAX(LEN({q})) AS [{name}__len_max]")
    select = ",\n    ".join(parts)
    return f"""SELECT\n    {select}\nFROM {schema}.{table}"""


def _build_databricks_stats_sql(cols: list[dict], catalog: str, schema: str, table: str) -> str:
    """Build a Databricks SQL query that computes per-column summary stats."""
    parts = []
    for col in cols:
        name, btype = col["name"], col["base_type"]
        q = f"`{name}`"
        parts.append(f"SUM(CASE WHEN {q} IS NULL THEN 1 ELSE 0 END) AS `{name}__nulls`")
        parts.append(f"COUNT(DISTINCT {q}) AS `{name}__distinct`")
        if btype in NUMERIC_TYPES:
            parts.append(f"MIN(CAST({q} AS DOUBLE)) AS `{name}__min`")
            parts.append(f"MAX(CAST({q} AS DOUBLE)) AS `{name}__max`")
            parts.append(f"AVG(CAST({q} AS DOUBLE)) AS `{name}__avg`")
        elif btype in DATE_TYPES:
            parts.append(f"MIN({q}) AS `{name}__min`")
            parts.append(f"MAX({q}) AS `{name}__max`")
        elif btype in STRING_TYPES:
            parts.append(f"MIN(LENGTH({q})) AS `{name}__len_min`")
            parts.append(f"MAX(LENGTH({q})) AS `{name}__len_max`")
    select = ",\n    ".join(parts)
    return f"SELECT\n    {select}\nFROM {catalog}.{schema}.{table}"


def _parse_stats_row(row: dict, cols: list[dict], side: str) -> list[dict]:
    """Parse a single-row stats result into per-column dicts."""
    results = []
    for col in cols:
        name, btype = col["name"], col["base_type"]
        entry = {
            "column": name,
            "type": btype,
            f"{side}_nulls": row.get(f"{name}__nulls"),
            f"{side}_distinct": row.get(f"{name}__distinct"),
        }
        if btype in NUMERIC_TYPES:
            entry[f"{side}_min"] = row.get(f"{name}__min")
            entry[f"{side}_max"] = row.get(f"{name}__max")
            entry[f"{side}_avg"] = row.get(f"{name}__avg")
        elif btype in DATE_TYPES:
            entry[f"{side}_min"] = str(row.get(f"{name}__min"))
            entry[f"{side}_max"] = str(row.get(f"{name}__max"))
        elif btype in STRING_TYPES:
            entry[f"{side}_len_min"] = row.get(f"{name}__len_min")
            entry[f"{side}_len_max"] = row.get(f"{name}__len_max")
        results.append(entry)
    return results


def compare_synapse_vs_databricks(
    databricks_conn,
    synapse_conn,
    synapse_schema_name: str,
    synapse_table_name: str,
    databricks_catalog_name: str,
    databricks_schema_name: str,
    databricks_table_name: str,
) -> dict:
    """
    Compare a Synapse table against a Databricks table.

    Performs (all via SQL — no row data downloaded):
      1. Schema comparison (column names + types)
      2. Row count comparison (with OMD filters on Synapse)
      3. Per-column summary statistics (nulls, distinct, min/max/avg)

    Returns a dict with comparison results and DataFrames.
    """

    # ── 1. Schema metadata ──
    synapse_schema_df = pl.read_database(f"""
        SELECT upper(column_name) AS column_name,
               upper(data_type)   AS synapse_data_type
        FROM information_schema.columns
        WHERE table_schema = '{synapse_schema_name}'
          AND table_name   = '{synapse_table_name}'
          AND column_name NOT LIKE 'OMD%'
          AND column_name NOT LIKE '%_SK'
        ORDER BY ordinal_position
    """, synapse_conn)

    databricks_schema_df = pl.read_database(f"""
        SELECT upper(column_name) AS column_name,
               upper(data_type)   AS databricks_data_type
        FROM {databricks_catalog_name}.information_schema.columns
        WHERE table_schema = '{databricks_schema_name}'
          AND table_name   = '{databricks_table_name}'
          AND column_name NOT LIKE 'rtlh%'
    """, databricks_conn)

    databricks_schema_df = databricks_schema_df.with_columns(
        pl.col("databricks_data_type")
        .map_elements(lambda x: DATABRICKS_TO_SYNAPSE_TYPE_MAP.get(base_type(x)), return_dtype=pl.Utf8)
        .alias("synapse_expected_type")
    )

    # ── 2. Schema comparison ──
    schema_comparison_df = synapse_schema_df.join(databricks_schema_df, on="column_name", how="full")

    schema_mismatches_df = schema_comparison_df.filter(
        pl.col("synapse_data_type").is_null()
        | pl.col("databricks_data_type").is_null()
        | (
            pl.col("synapse_data_type").map_elements(base_type, return_dtype=pl.Utf8)
            != pl.col("synapse_expected_type").map_elements(base_type, return_dtype=pl.Utf8)
        )
    )
    schema_matches = schema_mismatches_df.height == 0

    # ── 3. Row counts ──
    synapse_row_count = int(pl.read_database(f"""
        SELECT count(*) AS row_count
        FROM {synapse_schema_name}.{synapse_table_name}
    """, synapse_conn)["row_count"][0])

    databricks_row_count = int(pl.read_database(f"""
        SELECT count(*) AS row_count
        FROM {databricks_catalog_name}.{databricks_schema_name}.{databricks_table_name}
    """, databricks_conn)["row_count"][0])

    row_count_matches = synapse_row_count == databricks_row_count
    row_diff = databricks_row_count - synapse_row_count
    row_diff_pct = (100.0 * row_diff / synapse_row_count) if synapse_row_count else 0.0

    # ── 4. Per-column summary statistics (via SQL) ──
    common_cols = sorted(
        set(synapse_schema_df["column_name"].to_list())
        & set(databricks_schema_df["column_name"].to_list())
    )

    stats_comparison_df = pl.DataFrame([])
    synapse_stats_df = pl.DataFrame([])
    databricks_stats_df = pl.DataFrame([])

    if common_cols:
        # Build column info using Synapse types for common cols
        col_infos = []
        for c in common_cols:
            syn_row = synapse_schema_df.filter(pl.col("column_name") == c)
            if syn_row.height > 0:
                btype = base_type(syn_row["synapse_data_type"][0])
            else:
                btype = "STRING"
            col_infos.append({"name": c, "base_type": btype})

        # Query stats in batches to avoid Synapse "nested too deeply" error
        BATCH_SIZE = 10
        syn_row_merged = {}
        dbx_row_merged = {}

        for batch_start in range(0, len(col_infos), BATCH_SIZE):
            batch = col_infos[batch_start : batch_start + BATCH_SIZE]

            syn_sql = _build_synapse_stats_sql(batch, synapse_schema_name, synapse_table_name)
            dbx_sql = _build_databricks_stats_sql(
                batch, databricks_catalog_name, databricks_schema_name, databricks_table_name
            )

            syn_batch = pl.read_database(syn_sql, synapse_conn)
            dbx_batch = pl.read_database(dbx_sql, databricks_conn)

            if syn_batch.height > 0:
                syn_row_merged.update(syn_batch.row(0, named=True))
            if dbx_batch.height > 0:
                dbx_row_merged.update(dbx_batch.row(0, named=True))

        if syn_row_merged and dbx_row_merged:
            syn_parsed = _parse_stats_row(syn_row_merged, col_infos, "synapse")
            dbx_parsed = _parse_stats_row(dbx_row_merged, col_infos, "databricks")

            synapse_stats_df = pl.DataFrame(syn_parsed)
            databricks_stats_df = pl.DataFrame(dbx_parsed)

            stats_comparison_df = synapse_stats_df.join(
                databricks_stats_df.drop("type"), on="column", how="inner"
            )

    return {
        "schema_matches": schema_matches,
        "row_count_matches": row_count_matches,
        "synapse_row_count": synapse_row_count,
        "databricks_row_count": databricks_row_count,
        "row_diff": row_diff,
        "row_diff_pct": round(row_diff_pct, 2),
        "schema_comparison_df": schema_comparison_df,
        "schema_mismatches_df": schema_mismatches_df,
        "common_columns": common_cols,
        "stats_comparison_df": stats_comparison_df,
        "synapse_stats_df": synapse_stats_df,
        "databricks_stats_df": databricks_stats_df,
    }
