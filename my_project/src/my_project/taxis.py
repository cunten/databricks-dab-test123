from databricks.sdk.runtime import spark
from pyspark.sql import DataFrame
import pyspark.sql.functions as F


def find_all_taxis() -> DataFrame:
    """Find all taxi data."""
    return spark.read.table("samples.nyctaxi.trips")


def filter_long_trips(df: DataFrame, min_distance: float) -> DataFrame:
    """Return only trips with trip_distance >= min_distance."""
    return df.filter(F.col("trip_distance") >= min_distance)


def add_fare_per_mile(df: DataFrame) -> DataFrame:
    """Add a fare_per_mile column (fare_amount / trip_distance).
    Trips with zero distance get NULL.
    """
    return df.withColumn(
        "fare_per_mile",
        F.when(
            F.col("trip_distance") > 0, F.col("fare_amount") / F.col("trip_distance")
        ).otherwise(None),
    )


def drop_incomplete_rows(df: DataFrame) -> DataFrame:
    """Drop rows where fare_amount or trip_distance is NULL."""
    return df.dropna(subset=["fare_amount", "trip_distance"])
