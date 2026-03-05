import os

import pandas as pd
import plotly.express as px
import streamlit as st
from databricks import sql
from databricks.sdk.core import Config

st.set_page_config(
    page_title="NYC Taxi Explorer",
    page_icon="🚕",
    layout="wide",
)

st.title("🚕 NYC Taxi Data Explorer")
st.caption("Powered by `samples.nyctaxi.trips` · Databricks Apps")


@st.cache_resource(ttl=300)
def get_connection():
    cfg = Config()
    warehouse_id = os.environ["DATABRICKS_WAREHOUSE_ID"]
    return sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{warehouse_id}",
        credentials_provider=lambda: cfg.authenticate,
    )


@st.cache_data(ttl=600, show_spinner="Querying NYC Taxi data…")
def load_data(limit: int = 50_000) -> pd.DataFrame:
    conn = get_connection()
    query = f"""
        SELECT
            tpep_pickup_datetime,
            tpep_dropoff_datetime,
            passenger_count,
            trip_distance,
            fare_amount,
            tip_amount,
            total_amount,
            payment_type,
            HOUR(tpep_pickup_datetime)  AS pickup_hour,
            DATE(tpep_pickup_datetime)  AS pickup_date,
            DAYOFWEEK(tpep_pickup_datetime) AS day_of_week
        FROM samples.nyctaxi.trips
        WHERE fare_amount  > 0
          AND trip_distance > 0
          AND trip_distance < 100
          AND fare_amount  < 200
        LIMIT {limit}
    """
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    return pd.DataFrame(rows, columns=cols)


# ── Sidebar controls ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    row_limit = st.selectbox(
        "Sample size",
        options=[10_000, 25_000, 50_000],
        index=2,
        format_func=lambda x: f"{x:,} rows",
    )
    min_dist, max_dist = st.slider(
        "Trip distance (miles)",
        min_value=0.0,
        max_value=50.0,
        value=(0.0, 30.0),
        step=0.5,
    )
    min_fare, max_fare = st.slider(
        "Fare amount ($)",
        min_value=0,
        max_value=200,
        value=(0, 100),
        step=5,
    )

# ── Load & filter ────────────────────────────────────────────────────────────
df = load_data(limit=row_limit)

mask = df["trip_distance"].between(min_dist, max_dist) & df["fare_amount"].between(
    min_fare, max_fare
)
df_filtered = df[mask]

if df_filtered.empty:
    st.warning("No data matches the current filters. Adjust the sidebar controls.")
    st.stop()

# ── KPI row ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total trips", f"{len(df_filtered):,}")
k2.metric("Avg fare", f"${df_filtered['fare_amount'].mean():.2f}")
k3.metric("Avg distance", f"{df_filtered['trip_distance'].mean():.2f} mi")
k4.metric("Avg tip", f"${df_filtered['tip_amount'].mean():.2f}")

st.divider()

# ── Charts ───────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Fare distribution")
    fig_fare = px.histogram(
        df_filtered,
        x="fare_amount",
        nbins=50,
        labels={"fare_amount": "Fare ($)"},
        color_discrete_sequence=["#FF4B4B"],
    )
    fig_fare.update_layout(showlegend=False, margin=dict(t=20))
    st.plotly_chart(fig_fare, use_container_width=True)

with col_right:
    st.subheader("Trip distance distribution")
    fig_dist = px.histogram(
        df_filtered,
        x="trip_distance",
        nbins=50,
        labels={"trip_distance": "Distance (miles)"},
        color_discrete_sequence=["#1C83E1"],
    )
    fig_dist.update_layout(showlegend=False, margin=dict(t=20))
    st.plotly_chart(fig_dist, use_container_width=True)

col_left2, col_right2 = st.columns(2)

with col_left2:
    st.subheader("Fare vs. distance")
    fig_scatter = px.scatter(
        df_filtered.sample(min(3_000, len(df_filtered)), random_state=42),
        x="trip_distance",
        y="fare_amount",
        opacity=0.4,
        trendline="ols",
        labels={"trip_distance": "Distance (miles)", "fare_amount": "Fare ($)"},
        color_discrete_sequence=["#FF4B4B"],
    )
    fig_scatter.update_layout(margin=dict(t=20))
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_right2:
    st.subheader("Pickups by hour of day")
    hourly = (
        df_filtered.groupby("pickup_hour")
        .size()
        .reset_index(name="trip_count")
        .sort_values("pickup_hour")
    )
    fig_hour = px.bar(
        hourly,
        x="pickup_hour",
        y="trip_count",
        labels={"pickup_hour": "Hour of day", "trip_count": "Number of trips"},
        color_discrete_sequence=["#1C83E1"],
    )
    fig_hour.update_layout(showlegend=False, margin=dict(t=20))
    st.plotly_chart(fig_hour, use_container_width=True)

# ── Payment type breakdown ───────────────────────────────────────────────────
st.subheader("Payment type breakdown")
payment_map = {1: "Credit card", 2: "Cash", 3: "No charge", 4: "Dispute", 5: "Unknown"}
payment_counts = (
    df_filtered["payment_type"]
    .map(payment_map)
    .fillna("Other")
    .value_counts()
    .reset_index()
)
payment_counts.columns = ["payment_type", "count"]
fig_pie = px.pie(
    payment_counts,
    names="payment_type",
    values="count",
    hole=0.4,
)
fig_pie.update_layout(margin=dict(t=20))
st.plotly_chart(fig_pie, use_container_width=True)

# ── Raw data preview ─────────────────────────────────────────────────────────
with st.expander("Raw data preview (first 500 rows)"):
    st.dataframe(
        df_filtered.head(500),
        use_container_width=True,
        column_config={
            "fare_amount": st.column_config.NumberColumn("Fare ($)", format="$%.2f"),
            "tip_amount": st.column_config.NumberColumn("Tip ($)", format="$%.2f"),
            "total_amount": st.column_config.NumberColumn("Total ($)", format="$%.2f"),
            "trip_distance": st.column_config.NumberColumn(
                "Distance (mi)", format="%.2f"
            ),
        },
    )
