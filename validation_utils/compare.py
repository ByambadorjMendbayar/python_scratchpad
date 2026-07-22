"""Synapse vs Databricks data validation utilities.

Compares schema, row counts, and summary statistics via SQL queries.
No row-level data is downloaded.
"""

import polars as pl


DATABRICKS_TO_SYNAPSE_TYPE_MAP = {
    "TINYINT": ("TINYINT", "SMALLINT", "INT"),
    "SMALLINT": ("SMALLINT", "INT"),
    "SHORT": ("SMALLINT", "INT"),
    "INT": ("INT", "SMALLINT"),
    "INTEGER": ("INT", "SMALLINT"),
    "BIGINT": "BIGINT",
    "FLOAT": ("REAL", "FLOAT"),
    "REAL": ("REAL", "FLOAT"),
    "DOUBLE": ("FLOAT", "REAL"),
    "DECIMAL": ("DECIMAL", "NUMERIC"),
    "NUMERIC": ("NUMERIC", "DECIMAL"),
    "STRING": ("VARCHAR", "NVARCHAR", "VARBINARY"),
    "VARCHAR": ("VARCHAR", "NVARCHAR"),
    "CHAR": ("CHAR", "NCHAR"),
    "BINARY": ("VARBINARY", "BINARY"),
    "VARBINARY": ("VARBINARY", "BINARY"),
    "DATE": "DATE",
    "TIMESTAMP": ("DATETIME2", "DATETIME"),
    "TIMESTAMP_NTZ": ("DATETIME2", "DATETIME"),
    "TIMESTAMP_LTZ": ("DATETIMEOFFSET", "DATETIME2"),
    "BOOLEAN": ("BIT", "BOOLEAN"),
}

NUMERIC_TYPES = {"INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT", "FLOAT",
                 "REAL", "DOUBLE", "DECIMAL", "NUMERIC", "MONEY", "SMALLMONEY"}
DATE_TYPES = {"DATE", "DATETIME", "DATETIME2", "DATETIMEOFFSET", "SMALLDATETIME",
              "TIMESTAMP", "TIMESTAMP_NTZ", "TIMESTAMP_LTZ"}
STRING_TYPES = {"VARCHAR", "NVARCHAR", "CHAR", "NCHAR", "TEXT", "NTEXT", "STRING"}


def base_type(dtype: str) -> str:
    """Extract base type name, stripping precision/scale."""
    return str(dtype).split("(")[0].strip().upper() if dtype is not None else None


def _map_dbx_to_synapse_type(dbx_type: str) -> str:
    """Map a Databricks base type to the first matching Synapse type string."""
    mapped = DATABRICKS_TO_SYNAPSE_TYPE_MAP.get(dbx_type)
    if mapped is None:
        return None
    if isinstance(mapped, tuple):
        return mapped[0]  # return first for display; matching checks all
    return mapped


def _type_matches(synapse_type: str, dbx_type: str) -> bool:
    """Check if a Synapse type matches the expected Databricks→Synapse mapping."""
    if synapse_type is None or dbx_type is None:
        return False
    mapped = DATABRICKS_TO_SYNAPSE_TYPE_MAP.get(base_type(dbx_type))
    if mapped is None:
        return False
    syn_base = base_type(synapse_type)
    if isinstance(mapped, tuple):
        return syn_base in (base_type(m) for m in mapped)
    return syn_base == base_type(mapped)


def _build_synapse_stats_sql(cols: list[dict], schema: str, table: str) -> str:
    """Build a Synapse SQL query that computes per-column summary stats."""
    parts = []
    for col in cols:
        name = col["name"]  # normalized name (used as alias)
        syn_name = col.get("syn_name", name)  # original Synapse column name
        btype = col["base_type"]
        q = f"[{syn_name}]"
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
        .map_elements(lambda x: _map_dbx_to_synapse_type(base_type(x)), return_dtype=pl.Utf8)
        .alias("synapse_expected_type")
    )

    # ── 2. Schema comparison ──
    # Normalize Synapse column names: strip _RIO suffix for matching
    synapse_schema_df = synapse_schema_df.with_columns(
        pl.col("column_name")
        .str.replace(r"_RIO$", "")
        .alias("column_name_normalized")
    )

    schema_comparison_df = synapse_schema_df.join(
        databricks_schema_df,
        left_on="column_name_normalized",
        right_on="column_name",
        how="full",
    ).with_columns(
        # Use the normalized name as the canonical column name for display
        pl.coalesce("column_name_normalized", "column_name_right").alias("matched_column"),
    )

    schema_mismatches_df = schema_comparison_df.filter(
        pl.col("synapse_data_type").is_null()
        | pl.col("databricks_data_type").is_null()
        | ~pl.struct(["synapse_data_type", "databricks_data_type"]).map_elements(
            lambda s: _type_matches(s["synapse_data_type"], s["databricks_data_type"]),
            return_dtype=pl.Boolean,
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
    # Use normalized names to find common columns
    synapse_normalized = set(synapse_schema_df["column_name_normalized"].to_list())
    databricks_col_set = set(databricks_schema_df["column_name"].to_list())
    common_cols = sorted(synapse_normalized & databricks_col_set)

    # Build a mapping from normalized name → original Synapse column name
    syn_name_map = dict(zip(
        synapse_schema_df["column_name_normalized"].to_list(),
        synapse_schema_df["column_name"].to_list(),
    ))

    stats_comparison_df = pl.DataFrame([])
    synapse_stats_df = pl.DataFrame([])
    databricks_stats_df = pl.DataFrame([])

    if common_cols:
        # Build column info using Synapse types for common cols
        # syn_name is the original Synapse column name (may have _RIO suffix)
        # name is the normalized/Databricks column name
        col_infos = []
        for c in common_cols:
            syn_original = syn_name_map.get(c, c)
            syn_row = synapse_schema_df.filter(pl.col("column_name") == syn_original)
            if syn_row.height > 0:
                btype = base_type(syn_row["synapse_data_type"][0])
            else:
                btype = "STRING"
            col_infos.append({"name": c, "syn_name": syn_original, "base_type": btype})

        # Sample 15 random columns if table has too many (avoids Synapse nesting limit)
        import random
        MAX_STATS_COLS = 15
        if len(col_infos) > MAX_STATS_COLS:
            col_infos_sampled = sorted(random.sample(col_infos, MAX_STATS_COLS), key=lambda x: x["name"])
        else:
            col_infos_sampled = col_infos

        syn_stats_sql = _build_synapse_stats_sql(col_infos_sampled, synapse_schema_name, synapse_table_name)
        dbx_stats_sql = _build_databricks_stats_sql(
            col_infos_sampled, databricks_catalog_name, databricks_schema_name, databricks_table_name
        )

        syn_stats_raw = pl.read_database(syn_stats_sql, synapse_conn)
        dbx_stats_raw = pl.read_database(dbx_stats_sql, databricks_conn)

        if syn_stats_raw.height > 0 and dbx_stats_raw.height > 0:
            syn_row_merged = syn_stats_raw.row(0, named=True)
            dbx_row_merged = dbx_stats_raw.row(0, named=True)
            syn_parsed = _parse_stats_row(syn_row_merged, col_infos_sampled, "synapse")
            dbx_parsed = _parse_stats_row(dbx_row_merged, col_infos_sampled, "databricks")

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
