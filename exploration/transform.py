from pyspark.sql import functions as F
from transforms.api import transform, Input, Output, incremental, configure
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, TimestampType, DoubleType, StringType
import datetime
import re
from typing import List


def clean_column_name(col_name: str) -> str:
    """Clean column name by replacing special characters with underscores."""
    # Replace spaces and special characters with underscores
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', col_name)
    # Remove multiple consecutive underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    return cleaned


def add_cleaned_name_column(df):
    """Add a cleaned name column to the dataframe."""
    return df.withColumn(
        "name_cleaned",
        F.trim(
            F.regexp_replace(
                F.regexp_replace(F.col("name"), r'[^a-zA-Z0-9_]', '_'),
                r'_+', '_'
            )
        )
    )


def prepare_stream_data(clean_df, stream_type: str = "current"):
    """Prepare stream data by casting values and grouping by name and timestamp."""
    stream = (clean_df.dataframe(stream_type)
              .withColumn("value", F.col("value").cast(DoubleType()))
              .select("name", "timestamp", "value")
              .withColumn("timestamp", F.date_trunc("second", "timestamp"))
              .groupBy("name", "timestamp")
              .agg(F.collect_list("value").alias("values")))
    return stream


def create_pivoted_output(unpivoted_df, expected_tags=None):
    """Create pivoted output from unpivoted dataframe."""
    if expected_tags:
        return (unpivoted_df
                .groupBy("timestamp")
                .pivot("name_cleaned", expected_tags)
                .agg(F.any_value("median")))
    else:
        return (
            unpivoted_df
            .groupBy("timestamp")
            .pivot("name_cleaned")
            .agg(F.any_value("median"))
        )


def write_outputs(output_unpivoted, output_pivoted, output_column_map, 
                 unpivoted_df, pivoted_df, mode: str = "modify"):
    """Write all outputs with the specified mode."""
    output_unpivoted.set_mode(mode)
    output_unpivoted.write_dataframe(unpivoted_df)
    
    output_pivoted.set_mode(mode)
    output_pivoted.write_dataframe(pivoted_df)
    
    column_mapping_df = unpivoted_df.select("name", "name_cleaned").distinct()
    output_column_map.set_mode(mode)
    output_column_map.write_dataframe(column_mapping_df)


def process_overlapping_data(overlappable_prev_output, overlapped_df, added_stream, ctx):
    """Process data when there are overlaps between added and previous data."""
    # Update overlapped data with combined values
    overlapped_df = (overlapped_df
                    .withColumn("values", F.concat("values", "added_values"))
                    .transform(get_median)
                    .withColumn("forward_filled", F.lit(0)))
    
    # Keep non-overlapped previous data that wasn't forward filled
    prev_output_filtered = (overlappable_prev_output
                           .join(overlapped_df, on=["name", "timestamp"], how="leftanti")
                           .where(F.col("forward_filled") == 0)
                           .transform(add_cleaned_name_column))
    
    # Combine and forward fill
    combined_df = prev_output_filtered.unionByName(overlapped_df)
    return forward_fill(combined_df, ctx, ["values", "median"])


def process_non_overlapping_added_data(added_stream, ctx):
    """Process added data when there are no overlaps."""
    return (get_median(added_stream, "added_values")
            .transform(lambda df: forward_fill(df, ctx, ["values", "median"]))
            .transform(add_cleaned_name_column))


def process_incremental_data(ctx, output_unpivoted, output_pivoted, output_column_map, clean):
    """Handle incremental data processing."""
    # Get expected tags for pivoted output
    expected_tags = output_pivoted.dataframe("previous").columns
    expected_tags.remove("timestamp")
    
    # Prepare added stream data
    added_stream = prepare_stream_data(clean, "added")
    added_stream = added_stream.withColumn("added_values", F.col("values")).drop("values")
    
    # Find overlapping data
    window_threshold = added_stream.select(F.min("timestamp")).first()[0]
    overlappable_prev_output = (output_unpivoted.dataframe("previous")
                               .filter(F.col("timestamp") >= window_threshold))
    overlapped_df = overlappable_prev_output.join(added_stream, on=["name", "timestamp"], how="inner")
    
    if not overlapped_df.rdd.isEmpty():
        # Process overlapping data
        added_output_unpivoted = process_overlapping_data(
            overlappable_prev_output, overlapped_df, added_stream, ctx
        )
        
        # Union with historical data
        prev_output_unpivoted = (output_unpivoted.dataframe("previous")
                                .filter(F.col("timestamp") < window_threshold))
        final_output_unpivoted = (prev_output_unpivoted
                                 .unionByName(added_output_unpivoted)
                                 .transform(add_cleaned_name_column))
        
        # Create pivoted output
        added_output_pivoted = create_pivoted_output(added_output_unpivoted, expected_tags)
        prev_output_pivoted = (output_pivoted.dataframe("previous")
                              .filter(F.col("timestamp") < window_threshold))
        final_output_pivoted = prev_output_pivoted.unionByName(added_output_pivoted)
        
        # Write outputs with replace mode
        write_outputs(output_unpivoted, output_pivoted, output_column_map,
                     final_output_unpivoted, final_output_pivoted, "replace")
        
    elif not added_stream.rdd.isEmpty():
        # Process non-overlapping added data
        added_output_unpivoted = process_non_overlapping_added_data(added_stream, ctx)
        added_output_pivoted = create_pivoted_output(added_output_unpivoted, expected_tags)
        
        # Write outputs with modify mode
        write_outputs(output_unpivoted, output_pivoted, output_column_map,
                     added_output_unpivoted, added_output_pivoted, "modify")


def process_full_data(ctx, output_unpivoted, output_pivoted, output_column_map, clean):
    """Handle full data processing (non-incremental)."""
    # Prepare current stream data
    curr_stream = prepare_stream_data(clean, "current")
    
    # Process data
    curr_output_unpivoted = (get_median(curr_stream)
                            .transform(lambda df: forward_fill(df, ctx, ["values", "median"]))
                            .transform(add_cleaned_name_column))
    
    curr_output_pivoted = create_pivoted_output(curr_output_unpivoted)
    
    # Write outputs
    write_outputs(output_unpivoted, output_pivoted, output_column_map,
                 curr_output_unpivoted, curr_output_pivoted, "modify")


@configure(profile=['DYNAMIC_ALLOCATION_ENABLED_4_8'])
@incremental(semantic_version=2)
@transform(
    output_unpivoted=Output("ri.foundry.main.dataset.c1326935-6302-427f-933c-92a117a81749"),
    output_pivoted=Output("ri.foundry.main.dataset.a39cc5bc-22e9-4e71-942b-244b40650a21"),
    output_column_map=Output("/RioTinto/[Sandbox] Value Chain & Metrics/Value Chain and Insight Team/PP/Data/PI/Column Mapping"),
    clean=Input("ri.foundry.main.dataset.4dbfda13-b136-45b3-b452-5050683e27b0"),
)
def compute(ctx, output_unpivoted, output_pivoted, output_column_map, clean):
    """
    Main compute function that processes PI data in either incremental or full mode.
    
    In incremental mode:
    - Handles overlapping data between new and existing records
    - Merges values and recalculates medians for overlapping timestamps
    - Forward fills missing values
    
    In full mode:
    - Processes all current data from scratch
    - Calculates medians and forward fills missing values
    """
    if ctx.is_incremental:
        process_incremental_data(ctx, output_unpivoted, output_pivoted, output_column_map, clean)
    else:
        process_full_data(ctx, output_unpivoted, output_pivoted, output_column_map, clean)


def get_median(df, col: str = "values"):
    """
    Calculate median values from an array column.
    
    Args:
        df: Input dataframe
        col: Column name containing array of values
        
    Returns:
        Dataframe with median calculated and original values sorted
    """
    median_expr = """
        CASE
            WHEN n % 2 = 1 THEN values[CAST(n / 2 AS INT)]
            ELSE (values[CAST(n / 2 - 1 AS INT)] + values[CAST(n / 2 AS INT)]) / 2.0
        END
    """
    
    return (df
            .withColumn("values", F.array_sort(col))
            .withColumn("n", F.array_size("values"))
            .withColumn("median", F.expr(median_expr))
            .select("name", "timestamp", "values", "median"))


def forward_fill(df, ctx, cols: List[str]):
    """
    Forward fill missing values for specified columns using the last known value.
    
    Args:
        df: Input dataframe with timestamp and name columns
        ctx: Transform context containing spark session
        cols: List of column names to forward fill
        
    Returns:
        Dataframe with missing values forward filled
    """
    # Generate complete timestamp range for all tags
    range_df = generate_timestamps_tags_df(df, ctx)
    
    # Join with original data to identify missing values
    df_filled = range_df.join(df, on=["name", "timestamp"], how="left")
    
    # Create window for forward filling within each tag
    window = Window.partitionBy("name").orderBy("timestamp").rowsBetween(Window.unboundedPreceding, 0)
    
    # Mark forward filled rows
    df_filled = df_filled.withColumn(
        "forward_filled", 
        F.when(F.col(cols[0]).isNull(), 1).otherwise(0)
    )
    
    # Forward fill specified columns
    filled_exprs = [F.last(F.col(c), ignorenulls=True).over(window).alias(c) for c in cols]
    untouched_cols = [c for c in df_filled.columns if c not in cols]
    
    return df_filled.select(*untouched_cols, *filled_exprs)


def generate_timestamps_tags_df(df, ctx):
    """
    Generate a complete timestamp range for all tags in the dataset.
    
    For large time ranges (>12 hours), processes data in chunks to avoid memory issues.
    
    Args:
        df: Input dataframe with timestamp and name columns
        ctx: Transform context containing spark session
        
    Returns:
        Dataframe with all timestamp-tag combinations
    """
    CHUNK_SIZE_SECONDS = 3600  # Process in hourly chunks
    INTERVAL = "interval 1 second"
    CHUNK_THRESHOLD_HOURS = 12
    
    # Get time range and tag list
    start_ts = df.select(F.min("timestamp")).first()[0]
    end_ts = df.select(F.max("timestamp")).first()[0]
    tags_list = [row["name"] for row in df.select("name").distinct().collect()]
    tags_array = F.array([F.lit(tag) for tag in tags_list])
    
    total_hours = (end_ts - start_ts).total_seconds() / CHUNK_SIZE_SECONDS
    
    if total_hours < CHUNK_THRESHOLD_HOURS:
        # Process all at once for small time ranges
        return _generate_single_range(ctx, start_ts, end_ts, tags_array, INTERVAL)
    else:
        # Process in chunks for large time ranges
        return _generate_chunked_range(ctx, start_ts, end_ts, tags_array, INTERVAL, CHUNK_SIZE_SECONDS)


def _generate_single_range(ctx, start_ts, end_ts, tags_array, interval):
    """Generate timestamp range in a single operation."""
    start_end_df = ctx.spark_session.createDataFrame([(start_ts, end_ts)], ["start", "end"])
    return (start_end_df
            .select(F.explode(F.sequence("start", "end", F.expr(interval))).alias("timestamp"))
            .withColumn("name", F.explode(tags_array)))


def _generate_chunked_range(ctx, start_ts, end_ts, tags_array, interval, chunk_size):
    """Generate timestamp range in chunks to handle large time ranges."""
    # Create time chunks
    chunks = []
    current = start_ts
    while current < end_ts:
        chunk_end = min(current + datetime.timedelta(seconds=chunk_size), end_ts)
        chunks.append((current, chunk_end))
        current = chunk_end
    
    # Create dataframe of chunks
    schema = StructType([
        StructField("chunk_start", TimestampType(), False),
        StructField("chunk_end", TimestampType(), False)
    ])
    chunks_df = ctx.spark_session.createDataFrame(chunks, schema)
    
    # Generate timestamps for all chunks
    return (chunks_df
            .select(F.explode(F.sequence("chunk_start", "chunk_end", F.expr(interval))).alias("timestamp"))
            .withColumn("name", F.explode(tags_array)))

