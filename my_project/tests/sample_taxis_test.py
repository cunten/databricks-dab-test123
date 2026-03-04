from databricks.sdk.runtime import spark
from pyspark.sql import DataFrame
from my_project import taxis
from chispa import assert_df_equality


def test_find_all_taxis():
    results = taxis.find_all_taxis()
    assert results.count() > 5


# ── Unit tests using chispa ──────────────────────────────────────────────────


def test_filter_long_trips(spark):
    data = [
        (1, 1.5, 10.0),
        (2, 3.0, 15.0),
        (3, 0.8, 5.0),
        (4, 5.0, 25.0),
    ]
    df = spark.createDataFrame(data, ["id", "trip_distance", "fare_amount"])

    result = taxis.filter_long_trips(df, min_distance=2.0)
    expected = spark.createDataFrame(
        [(2, 3.0, 15.0), (4, 5.0, 25.0)],
        ["id", "trip_distance", "fare_amount"],
    )

    assert_df_equality(result, expected, ignore_row_order=True)


def test_add_fare_per_mile(spark):
    data = [
        (1, 2.0, 10.0),  # fare_per_mile = 5.0
        (2, 0.0, 5.0),  # zero distance → NULL
        (3, 4.0, 20.0),  # fare_per_mile = 5.0
    ]
    df = spark.createDataFrame(data, ["id", "trip_distance", "fare_amount"])

    result = taxis.add_fare_per_mile(df).select("id", "fare_per_mile")
    expected = spark.createDataFrame(
        [(1, 5.0), (2, None), (3, 5.0)],
        ["id", "fare_per_mile"],
    )

    assert_df_equality(result, expected, ignore_row_order=True)


def test_drop_incomplete_rows(spark):
    data = [
        (1, 2.0, 10.0),
        (2, None, 10.0),
        (3, 1.0, None),
        (4, 3.0, 15.0),
    ]
    df = spark.createDataFrame(data, ["id", "trip_distance", "fare_amount"])

    result = taxis.drop_incomplete_rows(df)
    expected = spark.createDataFrame(
        [(1, 2.0, 10.0), (4, 3.0, 15.0)],
        ["id", "trip_distance", "fare_amount"],
    )

    assert_df_equality(result, expected, ignore_row_order=True)
