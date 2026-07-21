"""Synapse vs Databricks data validation utilities."""

import polars as pl


# ─── Type constants ────────────────────────────────────────────────────────────

INT_LIKE = {
    pl.Boolean, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
}
FLOAT_LIKE = {pl.Float32, pl.Float64}
NUMERIC_LIKE = INT_LIKE | FLOAT_LIKE

DATABRICKS_TO_SYNAPSE_TYPE_MAP = {
    "TINYINT": "TINYINT",
    "SMALLINT": "SMALLINT",
    "INT": "INT",
    "INTEGER": "INT",
    "BIGINT": "BIGINT",
    "FLOAT": "REAL",
    "REAL": "REAL",
    "DOUBLE": "FLOAT",
    "DECIMAL": "DECIMAL",
    "NUMERIC": "NUMERIC",
    "STRING": "VARCHAR",
    "VARCHAR": "VARCHAR",
    "CHAR": "CHAR",
    "BINARY": "VARBINARY",
    "DATE": "DATE",
    "TIMESTAMP": "DATETIME2",
    "TIMESTAMP_NTZ": "DATETIME2",
    "TIMESTAMP_LTZ": "DATETIMEOFFSET",
    "BOOLEAN": "BIT",
}


# ─── Helper functions ──────────────────────────────────────────────────────────

def base_type(dtype: str) -> str:
    """Extract base type name (strip precision/scale)."""
    return str(dtype).split("(")[0].strip().upper() if dtype is not None else None


def normalize_types(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize integer/float/datetime types for consistent hashing."""
    exprs = []
    for c, d in df.schema.items():
        if d in INT_LIKE:
            exprs.append(pl.col(c).cast(pl.Int64).alias(c))
        elif d in FLOAT_LIKE:
            exprs.append(pl.col(c).cast(pl.Float64).alias(c))
        elif str(d).startswith("Datetime"):
            e = pl.col(c)
            if "time_zone" in str(d):
                e = e.dt.replace_time_zone(None)
            exprs.append(e.dt.cast_time_unit("us").alias(c))
    return df.with_columns(exprs) if exprs else df


def add_row_hash(df: pl.DataFrame) -> pl.DataFrame:
    """Add a row_hash column computed from all columns."""
    return df.with_columns(pl.struct(pl.all()).hash().alias("row_hash"))


def quantile_or_none(series: pl.Series, q: float):
    """Safely compute a quantile, returning None on empty series."""
    if series.len() == 0:
        return None
    try:
        return series.quantile(q, interpolation="nearest")
    except Exception:
        return series.quantile(q)


def mode_or_none(series: pl.Series):
    """Safely compute mode, returning None on empty series."""
    if series.len() == 0:
        return None
    mode_values = series.mode()
    if mode_values.len() == 0:
        return None
    return mode_values[0]


def calc_hash_match_stats(left_df: pl.DataFrame, right_df: pl.DataFrame) -> dict:
    """Compare two DataFrames by row hash and return match statistics."""
    left_h = add_row_hash(left_df)
    right_h = add_row_hash(right_df)

    left_counts = (
        left_h.group_by("row_hash").len().rename({"len": "left_cnt"})
    )
    right_counts = (
        right_h.group_by("row_hash").len().rename({"len": "right_cnt"})
    )

    cmp_df = (
        left_counts.join(right_counts, on="row_hash", how="full")
        .with_columns(
            pl.col("left_cnt").fill_null(0).cast(pl.Int64),
            pl.col("right_cnt").fill_null(0).cast(pl.Int64),
        )
        .with_columns(pl.min_horizontal("left_cnt", "right_cnt").alias("matched_cnt"))
    )

    left_total = int(left_counts["left_cnt"].sum()) if left_counts.height else 0
    right_total = int(right_counts["right_cnt"].sum()) if right_counts.height else 0
    matched_rows_local = int(cmp_df["matched_cnt"].sum()) if cmp_df.height else 0

    return {
        "left_rows": left_total,
        "right_rows": right_total,
        "matched_rows": matched_rows_local,
        "exact_full_hash_match": left_h["row_hash"].sort().equals(right_h["row_hash"].sort()),
    }


def build_numeric_profile(df: pl.DataFrame, cols: list[str], side: str) -> pl.DataFrame:
    """Build a statistical profile for numeric columns."""
    rows = []
    for c in cols:
        s = df[c].drop_nulls()
        rows.append({
            "column": c,
            f"{side}_min": s.min() if s.len() else None,
            f"{side}_q1": quantile_or_none(s, 0.25),
            f"{side}_median": s.median() if s.len() else None,
            f"{side}_q3": quantile_or_none(s, 0.75),
            f"{side}_max": s.max() if s.len() else None,
            f"{side}_mean": s.mean() if s.len() else None,
            f"{side}_std": s.std() if s.len() else None,
            f"{side}_null_count": int(df[c].null_count()),
        })
    return pl.DataFrame(rows) if rows else pl.DataFrame([])


def build_string_profile(df: pl.DataFrame, cols: list[str], side: str) -> pl.DataFrame:
    """Build a profile for string columns (lexical stats, length distribution)."""
    rows = []
    for c in cols:
        s = df[c].cast(pl.Utf8)
        non_null = s.drop_nulls()
        lengths = non_null.str.len_chars() if non_null.len() else pl.Series([], dtype=pl.Int64)
        rows.append({
            "column": c,
            f"{side}_lex_min": non_null.min() if non_null.len() else None,
            f"{side}_lex_max": non_null.max() if non_null.len() else None,
            f"{side}_mode": mode_or_none(non_null),
            f"{side}_n_unique": int(non_null.n_unique()) if non_null.len() else 0,
            f"{side}_null_count": int(s.null_count()),
            f"{side}_len_min": lengths.min() if lengths.len() else None,
            f"{side}_len_q1": quantile_or_none(lengths, 0.25),
            f"{side}_len_median": lengths.median() if lengths.len() else None,
            f"{side}_len_q3": quantile_or_none(lengths, 0.75),
            f"{side}_len_max": lengths.max() if lengths.len() else None,
        })
    return pl.DataFrame(rows) if rows else pl.DataFrame([])


# ─── Main comparison function ──────────────────────────────────────────────────

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

    Performs:
      1. Schema comparison (column names + types)
      2. Row count comparison (with OMD filters on Synapse)
      3. Date-based sampling for large tables (>100k rows)
      4. Full row-hash matching on common columns
      5. Numeric and string profiling

    Returns a dict with all comparison metrics and DataFrames.
    """

    # ── 1. Read column metadata ──
    synapse_schema_query = f"""
        select
            upper(column_name) as column_name,
            upper(data_type) as synapse_data_type
        from information_schema.columns
        where table_schema = '{synapse_schema_name}'
          and table_name = '{synapse_table_name}'
          and column_name not like 'OMD%'
          and column_name not like 'HSTG%'
        order by ordinal_position
    """

    databricks_schema_query = f"""
        select
            upper(column_name) as column_name,
            upper(data_type) as databricks_data_type
        from {databricks_catalog_name}.information_schema.columns
        where table_schema = '{databricks_schema_name}'
          and table_name = '{databricks_table_name}'
          and column_name not like 'rtlh%'
    """

    synapse_schema_df = pl.read_database(synapse_schema_query, synapse_conn)
    databricks_schema_df = pl.read_database(databricks_schema_query, databricks_conn)

    databricks_schema_df = databricks_schema_df.with_columns(
        pl.col("databricks_data_type")
        .map_elements(
            lambda x: DATABRICKS_TO_SYNAPSE_TYPE_MAP.get(base_type(x), None),
            return_dtype=pl.Utf8,
        )
        .alias("synapse_expected_type")
    )

    # ── 2. Compare schema ──
    schema_comparison_df = synapse_schema_df.join(
        databricks_schema_df, on="column_name", how="full"
    )

    schema_mismatches_df = schema_comparison_df.filter(
        pl.col("synapse_data_type").is_null()
        | pl.col("databricks_data_type").is_null()
        | (
            pl.col("synapse_data_type").map_elements(base_type, return_dtype=pl.Utf8)
            != pl.col("synapse_expected_type").map_elements(base_type, return_dtype=pl.Utf8)
        )
    )

    schema_matches = schema_mismatches_df.height == 0

    # ── 3. Row count check ──
    synapse_count_query = f"""
        select count(*) as row_count
        from {synapse_schema_name}.{synapse_table_name}
        where omd_current_record_indicator = 'Y'
          and omd_deleted_record_indicator = 'N'
    """

    databricks_count_query = f"""
        select count(*) as row_count
        from {databricks_catalog_name}.{databricks_schema_name}.{databricks_table_name}
    """

    synapse_row_count = int(pl.read_database(synapse_count_query, synapse_conn)["row_count"][0])
    databricks_row_count = int(pl.read_database(databricks_count_query, databricks_conn)["row_count"][0])
    row_count_matches = synapse_row_count == databricks_row_count

    # ── 4. Detect date columns and apply sampling for large tables ──
    date_type_keywords = ["DATE", "DATETIME", "TIMESTAMP"]
    synapse_date_cols = [
        c for c in synapse_schema_df["column_name"].to_list()
        if any(
            kw in synapse_schema_df.filter(pl.col("column_name") == c)["synapse_data_type"][0]
            for kw in date_type_keywords
        )
    ]
    databricks_date_cols = [
        c for c in databricks_schema_df["column_name"].to_list()
        if any(
            kw in databricks_schema_df.filter(pl.col("column_name") == c)["databricks_data_type"][0]
            for kw in date_type_keywords
        )
    ]

    sampling_needed = synapse_row_count > 100_000 or databricks_row_count > 100_000
    date_filter_clause_syn = ""
    date_filter_clause_dbx = ""
    synapse_date_range = None
    databricks_date_range = None
    sampled_row_count_syn = synapse_row_count
    sampled_row_count_dbx = databricks_row_count

    if sampling_needed and synapse_date_cols:
        synapse_date_col = synapse_date_cols[0]
        synapse_date_query = f"""
            select max([{synapse_date_col}]) as max_date, min([{synapse_date_col}]) as min_date
            from {synapse_schema_name}.{synapse_table_name}
            where omd_current_record_indicator = 'Y'
              and omd_deleted_record_indicator = 'N'
        """
        synapse_date_result = pl.read_database(synapse_date_query, synapse_conn)
        synapse_max_date = synapse_date_result["max_date"][0]
        synapse_min_date = synapse_date_result["min_date"][0]
        synapse_date_range = (synapse_min_date, synapse_max_date)

        if synapse_max_date is not None:
            synapse_cutoff = f"DATEADD(month, -1, '{synapse_max_date}')"
            date_filter_clause_syn = f" and [{synapse_date_col}] >= {synapse_cutoff}"

            synapse_sampled_count_query = f"""
                select count(*) as row_count
                from {synapse_schema_name}.{synapse_table_name}
                where omd_current_record_indicator = 'Y'
                  and omd_deleted_record_indicator = 'N'
                  and [{synapse_date_col}] >= {synapse_cutoff}
            """
            sampled_row_count_syn = int(
                pl.read_database(synapse_sampled_count_query, synapse_conn)["row_count"][0]
            )

    if sampling_needed and databricks_date_cols:
        databricks_date_col = databricks_date_cols[0]
        databricks_date_query = f"""
            select max(`{databricks_date_col}`) as max_date, min(`{databricks_date_col}`) as min_date
            from {databricks_catalog_name}.{databricks_schema_name}.{databricks_table_name}
        """
        databricks_date_result = pl.read_database(databricks_date_query, databricks_conn)
        databricks_max_date = databricks_date_result["max_date"][0]
        databricks_min_date = databricks_date_result["min_date"][0]
        databricks_date_range = (databricks_min_date, databricks_max_date)

        if databricks_max_date is not None:
            date_filter_clause_dbx = (
                f" and `{databricks_date_col}` >= date_sub('{databricks_max_date}', 30)"
            )

            databricks_sampled_count_query = f"""
                select count(*) as row_count
                from {databricks_catalog_name}.{databricks_schema_name}.{databricks_table_name}
                where `{databricks_date_col}` >= date_sub('{databricks_max_date}', 30)
            """
            sampled_row_count_dbx = int(
                pl.read_database(databricks_sampled_count_query, databricks_conn)["row_count"][0]
            )

    # ── 5. Full match on common columns ──
    common_cols = sorted(
        set(synapse_schema_df["column_name"].to_list())
        & set(databricks_schema_df["column_name"].to_list())
    )

    empty_result = {
        "schema_matches": schema_matches,
        "row_count_matches": row_count_matches,
        "full_match": False,
        "reason": "No common columns found for full comparison.",
        "synapse_row_count": synapse_row_count,
        "databricks_row_count": databricks_row_count,
        "sampling_applied": sampling_needed,
        "sampled_row_count_synapse": sampled_row_count_syn,
        "sampled_row_count_databricks": sampled_row_count_dbx,
        "synapse_date_range": synapse_date_range,
        "databricks_date_range": databricks_date_range,
        "schema_mismatches_df": schema_mismatches_df,
        "schema_comparison_df": schema_comparison_df,
        "common_columns_compared": [],
        "matched_rows": 0,
        "syn_total": 0,
        "dbx_total": 0,
        "pct_of_synapse": 0.0,
        "pct_of_databricks": 0.0,
        "pct_overall": 0.0,
        "numeric_profile_comparison_df": pl.DataFrame([]),
        "string_profile_comparison_df": pl.DataFrame([]),
    }

    if not common_cols:
        return empty_result

    synapse_cols_sql = ", ".join([f"[{c}] as [SYNAPSE_{c}]" for c in common_cols])
    databricks_cols_sql = ", ".join([f"`{c}` as `DATABRICKS_{c}`" for c in common_cols])

    synapse_full_query = f"""
        select {synapse_cols_sql}
        from {synapse_schema_name}.{synapse_table_name}
        where omd_current_record_indicator = 'Y'
          and omd_deleted_record_indicator = 'N'
          {date_filter_clause_syn}
    """

    databricks_full_query = f"""
        select {databricks_cols_sql}
        from {databricks_catalog_name}.{databricks_schema_name}.{databricks_table_name}
        where 1=1
        {date_filter_clause_dbx}
    """

    synapse_full_table_df = pl.read_database(synapse_full_query, synapse_conn)
    databricks_full_table_df = pl.read_database(databricks_full_query, databricks_conn)

    # Strip timezone info from Databricks datetime columns
    tz_cols = [
        name for name, dtype in databricks_full_table_df.schema.items()
        if dtype == pl.Datetime and getattr(dtype, "time_zone", None) is not None
    ]
    if tz_cols:
        databricks_full_table_df = databricks_full_table_df.with_columns(
            [pl.col(c).dt.replace_time_zone(None) for c in tz_cols]
        )

    # Align column names
    syn_aligned = synapse_full_table_df.select(
        [pl.col(f"SYNAPSE_{c}").alias(c) for c in common_cols]
    )
    dbx_aligned = databricks_full_table_df.select(
        [pl.col(f"DATABRICKS_{c}").alias(c) for c in common_cols]
    )

    syn_norm = normalize_types(syn_aligned)
    dbx_norm = normalize_types(dbx_aligned)

    baseline_stats = calc_hash_match_stats(syn_norm, dbx_norm)
    full_match_exact = baseline_stats["exact_full_hash_match"]

    syn_total = baseline_stats["left_rows"]
    dbx_total = baseline_stats["right_rows"]
    matched_rows = min(baseline_stats["matched_rows"], syn_total, dbx_total)

    pct_of_synapse = 100.0 * matched_rows / syn_total if syn_total else 100.0
    pct_of_databricks = 100.0 * matched_rows / dbx_total if dbx_total else 100.0
    pct_overall = (
        100.0 * matched_rows / max(syn_total, dbx_total)
        if max(syn_total, dbx_total)
        else 100.0
    )

    # ── 6. Numeric and string profiling ──
    numeric_cols = [
        c for c in common_cols
        if syn_norm.schema.get(c) in NUMERIC_LIKE and dbx_norm.schema.get(c) in NUMERIC_LIKE
    ]
    string_cols = [
        c for c in common_cols
        if syn_norm.schema.get(c) == pl.Utf8 and dbx_norm.schema.get(c) == pl.Utf8
    ]

    syn_numeric_profile_df = build_numeric_profile(syn_norm, numeric_cols, "synapse")
    dbx_numeric_profile_df = build_numeric_profile(dbx_norm, numeric_cols, "databricks")
    if syn_numeric_profile_df.height and dbx_numeric_profile_df.height:
        numeric_profile_comparison_df = syn_numeric_profile_df.join(
            dbx_numeric_profile_df, on="column", how="inner"
        )
    else:
        numeric_profile_comparison_df = pl.DataFrame([])

    syn_string_profile_df = build_string_profile(syn_norm, string_cols, "synapse")
    dbx_string_profile_df = build_string_profile(dbx_norm, string_cols, "databricks")
    if syn_string_profile_df.height and dbx_string_profile_df.height:
        string_profile_comparison_df = syn_string_profile_df.join(
            dbx_string_profile_df, on="column", how="inner"
        )
    else:
        string_profile_comparison_df = pl.DataFrame([])

    return {
        "schema_matches": schema_matches,
        "row_count_matches": row_count_matches,
        "full_match": full_match_exact,
        "synapse_row_count": synapse_row_count,
        "databricks_row_count": databricks_row_count,
        "sampling_applied": sampling_needed,
        "sampled_row_count_synapse": sampled_row_count_syn,
        "sampled_row_count_databricks": sampled_row_count_dbx,
        "synapse_date_range": synapse_date_range,
        "databricks_date_range": databricks_date_range,
        "schema_mismatches_df": schema_mismatches_df,
        "schema_comparison_df": schema_comparison_df,
        "common_columns_compared": common_cols,
        "matched_rows": matched_rows,
        "syn_total": syn_total,
        "dbx_total": dbx_total,
        "pct_of_synapse": pct_of_synapse,
        "pct_of_databricks": pct_of_databricks,
        "pct_overall": pct_overall,
        "numeric_profile_comparison_df": numeric_profile_comparison_df,
        "string_profile_comparison_df": string_profile_comparison_df,
    }
