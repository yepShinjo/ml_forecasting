import pandas as pd
import numpy as np
import sqlalchemy
from prophet import Prophet
import pymysql   # for SHOW DATABASES
#from prophet.serialize import model_to_json, model_from_json  # Only needed if you want to save/load Prophet models
from datetime import datetime

# ---- 1. RDS Config ----
EXCLUDE_DBS = [
    'phpmyadmin', 'phpmyadmin2', 'horde', 'phppoint_forums', 'staging_site', 'sys',
    'roundcube', 'pos', 'bntennis_site', 'mysql', 'information_schema', 'performance_schema'
]
db_user = "admin"
db_password = "GUNGBUILDYEp_69"
db_host = "database-2.ccv8sgeuslw7.us-east-1.rds.amazonaws.com"
db_port = 3306

def ensure_forecast_table(engine):
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS phppos_sales_forecast (
        id INT AUTO_INCREMENT PRIMARY KEY,
        location_id VARCHAR(128),
        forecast_date DATE,
        avg_daily DOUBLE,
        max_daily DOUBLE,
        total_low DOUBLE,
        total_up DOUBLE,
        recommended_inventory DOUBLE
    );
    """
    with engine.connect() as conn:
        conn.execute(sqlalchemy.text(create_table_sql))

def write_forecast_to_db(engine, location, summary, forecast_date):
    insert_sql = """
    INSERT INTO phppos_sales_forecast 
        (location_id, forecast_date, avg_daily, max_daily, total_low, total_up, recommended_inventory)
    VALUES 
        (:location_id, :forecast_date, :avg_daily, :max_daily, :total_low, :total_up, :recommended_inventory)
    """
    with engine.begin() as conn:  # begin() auto-commits at the end
        conn.execute(
            sqlalchemy.text(insert_sql),
            {
                'location_id': location,
                'forecast_date': forecast_date,
                'avg_daily': summary['avg_daily'],
                'max_daily': summary['max_daily'],
                'total_low': summary['total_low'],
                'total_up': summary['total_up'],
                'recommended_inventory': summary['total_est']*0.95
            }
        )




# ---- 2. Helper to Get Database Names ----
def get_databases_to_process():
    databases = []
    try:
        conn = pymysql.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            port=db_port
        )
        with conn.cursor() as cur:
            cur.execute('SHOW DATABASES')
            for (db_name,) in cur.fetchall():
                if db_name not in EXCLUDE_DBS:
                    databases.append(db_name)
        conn.close()
    except Exception as ex:
        print(f"Error connecting to {db_host}: {ex}")
    return databases

# ---- 3. Prophet Forecast Function (as before) ----
def forecast_original(df_raw, periods=30, outlier_cap=30000, recent_months=3, dup_factor=3):
    df = df_raw.copy()
    df['sale_time'] = pd.to_datetime(df['sale_time'])

    daily = (df.groupby(df['sale_time'].dt.date)['total']
               .sum()
               .reset_index()
               .rename(columns={'sale_time': 'ds', 'total': 'y'}))
    daily['ds'] = pd.to_datetime(daily['ds'])
    daily = daily[daily['y'] < outlier_cap]

    latest = daily['ds'].max()
    nine_month_cut = latest - pd.DateOffset(months=9)
    df_9m = daily[daily['ds'] >= nine_month_cut]

    recent_cut = latest - pd.DateOffset(months=recent_months)
    df_recent = df_9m[df_9m['ds'] >= recent_cut]
    df_weighted = pd.concat([df_9m] + [df_recent] * dup_factor, ignore_index=True)

    m = Prophet(interval_width=0.85,
                daily_seasonality=False,
                changepoint_prior_scale=0.8,
                changepoint_range=0.98,
                seasonality_mode='multiplicative')
    m.fit(df_weighted)

    future = m.make_future_dataframe(periods=periods, freq='D')
    forecast = m.predict(future)
    fc_future = forecast[forecast['ds'] > df_weighted['ds'].max()][
        ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    ]

    summary = dict(
        avg_daily = fc_future['yhat'].mean(),
        max_daily = fc_future['yhat'].max(),
        total_est = fc_future['yhat'].sum(),
        total_low = fc_future['yhat_lower'].sum(),
        total_up  = fc_future['yhat_upper'].sum(),
        days      = len(fc_future)
    )
    return fc_future, summary

# ---- 4. Main Forecast Loop ----
def process_forecasts():
    dbs_to_process = get_databases_to_process()
    for db_name in dbs_to_process:
        print(f"\n--- Processing forecasts for DB: {db_name} ---")
        conn_str = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        engine = sqlalchemy.create_engine(conn_str)
        query = """
        SELECT sale_time, total, location_id
        FROM phppos_sales
        WHERE sale_time IS NOT NULL
        """
        try:
            df = pd.read_sql(query, engine)
        except Exception as ex:
            print(f"Could not query {db_name}: {ex}")
            continue
        if df.empty:
            print(f"No sales data found in {db_name}")
            continue
        df['total'] = pd.to_numeric(df['total'], errors='coerce').fillna(0)

        # Per-location forecasts
        location_ids = df['location_id'].unique()
        forecasts_original = {}
        summaries_original = {}
        for loc in location_ids:
            df_loc = df[df['location_id'] == loc]
            fc, sm = forecast_original(df_loc, periods=30)
            forecasts_original[loc] = fc
            summaries_original[loc] = sm
        # Total/all-location forecast
        fc_total, sm_total = forecast_original(df, periods=30)
        summaries_original['ALL'] = sm_total

        # Ensure forecast table exists!
        ensure_forecast_table(engine)
        today = pd.Timestamp.today().date()

        # Write results to the DB
        for loc, sm in summaries_original.items():
            loc_id = 'ALL' if loc == 'ALL' else str(loc)
            write_forecast_to_db(engine, loc_id, sm, today)

        # (Optional) Print summaries for reference
        for loc, sm in summaries_original.items():
            print(f"""
📍 Location {'ALL LOCATIONS' if loc=='ALL' else loc}
📊 Sales Forecast Summary (Next {sm['days']} Days):
- 📈 Average daily forecasted sales: ${sm['avg_daily']:,.0f}
- 🔺 Highest predicted daily sales: ${sm['max_daily']:,.0f}
- 💰 Total expected sales (best estimate): ${sm['total_est']:,.0f}
- 📉 Lower bound (cautious estimate): ${sm['total_low']:,.0f}
- 🔝 Upper bound (optimistic estimate): ${sm['total_up']:,.0f}
🛒 Recommended inventory plan: Prepare stock for around ${sm['total_est']*0.95:,.0f} in sales and monitor performance weekly.
""")

if __name__ == "__main__":
    process_forecasts()
