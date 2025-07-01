import pandas as pd
import numpy as np
from prophet import Prophet
import sqlalchemy
import pymysql
from sqlalchemy import text
import argparse

# ---------- 1. Generic helpers ----------

def ensure_column_exists(engine, table, column, dtype):
    with engine.begin() as conn:
        exists = conn.execute(text(
            """
            SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
            WHERE table_name=:tbl AND column_name=:col
            """
        ), {"tbl": table, "col": column}).scalar()
        if exists == 0:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {dtype}"))
            print(f"[SCHEMA] Added {column} to {table}")


def ensure_schema(engine):
    # Ensure operational columns in phppos_location_items exist
    ensure_column_exists(engine, "phppos_location_items", "forecasted_reorder_level", "INT DEFAULT NULL")
    ensure_column_exists(engine, "phppos_location_items", "forecasted_replenish_level", "INT DEFAULT NULL")


def upsert_forecasted_levels_for_items(results_df, engine):
    with engine.begin() as conn:
        for _, row in results_df.iterrows():
            conn.execute(sqlalchemy.text("""
                INSERT INTO phppos_location_items (
                    location_id,
                    item_id,
                    forecasted_reorder_level,
                    forecasted_replenish_level
                )
                VALUES (
                    :location_id,
                    :item_id,
                    :forecasted_reorder_level,
                    :forecasted_replenish_level
                )
                ON DUPLICATE KEY UPDATE
                    forecasted_reorder_level = :forecasted_reorder_level,
                    forecasted_replenish_level = :forecasted_replenish_level
            """), {
                "location_id": row['location_id'],
                "item_id": row['item_id'],
                "forecasted_reorder_level": row['forecasted_reorder_level'],
                "forecasted_replenish_level": row['forecasted_replenish_level']
            })


def write_results_to_db(results_df, engine):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS phppos_item_variation_forecasts (
        id INT AUTO_INCREMENT PRIMARY KEY,
        item_id INT,
        variation_id INT,
        location_id INT,
        forecasted_reorder_level INT,
        forecasted_replenish_level INT,
        enough_history BOOLEAN,
        z_score FLOAT,
        demand_lt FLOAT,
        sigma_lt FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (location_id)
            REFERENCES phppos_locations(location_id)
    );
    """
    with engine.connect() as conn:
        conn.execute(sqlalchemy.text(create_table_query))
    results_df['variation_id'] = None  # Explicitly set NULL for item-level rows
    results_df.to_sql(
        'phppos_item_variation_forecasts',
        engine,
        if_exists='append',
        index=False,
        method='multi'
    )

# ---------- 2. Item-level forecast ----------

# ONLY Top 200 items in the last 12 months will be forecasted
# Out of the 200 items IF : that item has been sold across atleast 20 days and 4 weeks -> THEN we use prophet else we use normal math



def run_item_forecast_for_database(conn_str, top_n=200):
    engine = sqlalchemy.create_engine(conn_str)
    item_sql = """
      SELECT date(sale_time) AS sale_date,
             location_id,
             item_id,
             SUM(quantity_purchased) AS qty
      FROM phppos_sales_items
      INNER JOIN phppos_sales USING(sale_id)
      WHERE quantity_purchased > 0
      GROUP BY sale_date, location_id, item_id
    """
    daily = pd.read_sql(item_sql, engine, parse_dates=['sale_date'])
    if daily.empty:
        print("[WARN] No sales found.")
        return pd.DataFrame()
    cutoff = daily['sale_date'].max() - pd.DateOffset(months=12)
    last_year = daily[daily['sale_date'] >= cutoff]
    top_items = (last_year.groupby('item_id')['qty']
                 .sum()
                 .nlargest(top_n)
                 .index.tolist())
    last_year = last_year[last_year['item_id'].isin(top_items)]

    results = []
    min_days, min_weeks, lead_days = 20, 4, 7
    for (loc, item), grp in last_year.groupby(['location_id', 'item_id']):
        grp = grp.sort_values('sale_date')
        hist = grp.rename(columns={'sale_date': 'ds', 'qty': 'y'})
        enough_days = hist['ds'].nunique() >= min_days
        enough_weeks = hist['ds'].dt.isocalendar().week.nunique() >= min_weeks
        enough = enough_days and enough_weeks

        reorder, replenish, sigma_lt, z_sel = 0, 0, 1, 1.65
        try:
            if enough:
                m = Prophet(daily_seasonality=True)
                m.fit(hist)
                fc = m.predict(m.make_future_dataframe(periods=lead_days)).tail(lead_days)
                demand_lt = fc['yhat'].sum()
                sigma_lt = (fc['yhat_upper'].sum() - fc['yhat_lower'].sum()) / 3.29
                cv = hist['y'].std() / hist['y'].mean() if hist['y'].mean() else 1
                z_sel = 1.65 if cv < 0.5 else (2.0 if cv < 1.0 else 2.33)
                reorder = int(np.round(demand_lt + z_sel * sigma_lt))
                replenish = int(np.round(reorder + demand_lt))
            else:
                demand_lt = hist.tail(7)['y'].mean() * lead_days
                reorder = int(np.round(demand_lt))
                replenish = int(np.round(demand_lt * 2))
        except Exception:
            demand_lt = hist.tail(7)['y'].mean() * lead_days
            reorder = int(np.round(demand_lt))
            replenish = int(np.round(demand_lt * 2))

        results.append({
            'location_id': loc,
            'item_id': item,
            'forecasted_reorder_level': reorder,
            'forecasted_replenish_level': replenish,
            'enough_history': enough,
            'z_score': z_sel,
            'demand_lt': demand_lt,
            'sigma_lt': sigma_lt
        })

    return pd.DataFrame(results)

# ---------- 3. DB discovery ----------

EXCLUDE_DBS = [
    'phpmyadmin', 'phpmyadmin2', 'horde', 'phppoint_forums', 'staging_site', 'sys',
    'roundcube', 'pos', 'bntennis_site', 'mysql', 'information_schema', 'performance_schema'
]

DB_SERVERS = [
    {'host': 'database-2.ccv8sgeuslw7.us-east-1.rds.amazonaws.com', 'user': 'admin', 'password': 'GUNGBUILDYEp_69'},
    {'host': 'database-3.ccv8sgeuslw7.us-east-1.rds.amazonaws.com', 'user': 'admin', 'password': 'GUNGBUILDYEp_69'}
]

def get_databases_to_process():
    databases = []
    for server in DB_SERVERS:
        try:
            conn = pymysql.connect(
                host=server['host'],
                user=server['user'],
                password=server['password']
            )
            with conn.cursor() as cur:
                cur.execute('SHOW DATABASES')
                for (db_name,) in cur.fetchall():
                    if db_name not in EXCLUDE_DBS:
                        databases.append(db_name.replace('staging', ''))
            conn.close()
        except Exception as ex:
            print(f"Error connecting to {server['host']}: {ex}")
    return databases

# ---------- 4. Orchestration ----------

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'db_arg',
        help="Use -1 for all DBs, a positive number (e.g. 5) for first N DBs, or a DB name for just that one"
    )
    args = parser.parse_args()
    arg = args.db_arg

    for server in DB_SERVERS:
        dbs_to_process = get_databases_to_process()
        if not dbs_to_process:
            print(f"No databases found on {server['host']}.")
            continue

        # Handle CLI argument
        try:
            num = int(arg)
            if num == -1:
                dbs_selected = dbs_to_process
            elif num > 0:
                dbs_selected = dbs_to_process[:num]
            else:
                print("Invalid number argument. Must be -1 or a positive number.")
                continue
        except ValueError:
            if arg in dbs_to_process:
                dbs_selected = [arg]
            else:
                print(f"Database '{arg}' not found in: {dbs_to_process}")
                continue

        # Forecast for each DB
        for db in dbs_selected:
            conn_str = f"mysql+pymysql://{server['user']}:{server['password']}@{server['host']}:{server.get('port', 3306)}/{db}"
            print(f"--- Forecasting for DB: {db} on {server['host']} ---")
            engine = sqlalchemy.create_engine(conn_str)

            try:
                ensure_schema(engine)

                item_df = run_item_forecast_for_database(conn_str, top_n=200)

                if item_df.empty:
                    print(f"[SKIPPED] No item sales for DB: {db}")
                    continue

                write_results_to_db(item_df, engine)
                upsert_forecasted_levels_for_items(item_df, engine)

                print(f"[DONE] Forecasting complete for {db}")

            except Exception as exc:
                print(f"[ERROR] {db}: {exc}")

if __name__ == "__main__":
    main()