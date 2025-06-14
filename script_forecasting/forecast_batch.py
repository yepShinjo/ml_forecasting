import pandas as pd
import numpy as np
from prophet import Prophet
import sqlalchemy
import pymysql

def run_forecast_for_database(conn_str, output_path=None):
    # Your forecasting code as in the notebook goes here
    engine = sqlalchemy.create_engine(conn_str)
    variations_df = pd.read_sql_query(
        """SELECT phppos_sales.sale_time, phppos_sales_items.quantity_purchased, phppos_items.name,GROUP_CONCAT(DISTINCT phppos_attributes.name, ": ", phppos_attribute_values.name SEPARATOR ", ") as variation_name, phppos_sales_items.sale_id, phppos_sales_items.item_id,phppos_sales_items.item_variation_id, phppos_sales.location_id, phppos_sales_items.total FROM phppos_sales_items INNER JOIN phppos_sales USING(sale_id) INNER JOIN phppos_items ON phppos_items.item_id = phppos_sales_items.item_id INNER JOIN phppos_item_variations ON phppos_item_variations.id= phppos_sales_items.item_variation_id INNER JOIN phppos_item_variation_attribute_values ON phppos_item_variation_attribute_values.item_variation_id = phppos_sales_items.item_variation_id
     INNER JOIN phppos_attribute_values ON phppos_item_variation_attribute_values.attribute_value_id = phppos_attribute_values.id 
    INNER JOIN phppos_attributes ON phppos_attributes.id = phppos_attribute_values.attribute_id GROUP BY phppos_sales_items.sale_id, phppos_sales_items.item_id,phppos_sales_items.item_variation_id""", engine
    )
    # ... (rest of your notebook code)
    variations_df['sale_date'] = pd.to_datetime(variations_df['sale_time']).dt.date

    variations_df.sample(n=5)

    agg_df = variations_df.groupby(
        ['sale_date', 'item_variation_id', 'location_id', 'variation_name', 'name']
    )['quantity_purchased'].sum().reset_index()

    # track when does those returns happening if want to print
    returns = agg_df[agg_df['quantity_purchased'] < 0]

    # exclude those negative (barang di return) value from df
    agg_df = agg_df[agg_df['quantity_purchased'] >= 0]

    recent_daily_var_sales = agg_df.rename(
        columns={
            'sale_date': 'date',
            'item_variation_id': 'variation_id',
            'quantity_purchased': 'y'
        }
    )

    # Make sure 'date' is datetime
    recent_daily_var_sales['date'] = pd.to_datetime(recent_daily_var_sales['date'])
    latest_date = recent_daily_var_sales['date'].max()
    cutoff_date = latest_date - pd.DateOffset(months=12)

    # Filter for the last 12 months
    recent_12m = recent_daily_var_sales[recent_daily_var_sales['date'] >= cutoff_date].copy()

    recent_12m['sales_week'] = recent_12m['date'].dt.isocalendar().week
    recent_12m['sales_year'] = recent_12m['date'].dt.isocalendar().year

    history_quality = (
        recent_12m.groupby(['location_id', 'variation_id'])
        .agg(
            num_days_with_sales=('date', 'nunique'),
            num_weeks_with_sales=('sales_week', 'nunique'),
            num_years_with_sales=('sales_year', 'nunique')  # Useful for data that spans New Year
        )
        .reset_index()
    )

    history_quality = history_quality.merge(
        recent_12m[['location_id', 'variation_id', 'variation_name', 'name']].drop_duplicates(),
        on=['location_id', 'variation_id'],
        how='left'
    )

    min_days = 20
    min_weeks = 4

    history_quality['enough_history'] = (
        (history_quality['num_days_with_sales'] >= min_days) &
        (history_quality['num_weeks_with_sales'] >= min_weeks)
    )

    demand_stats = (
        recent_12m.groupby(['location_id', 'variation_id'])['y']
        .agg(['mean', 'std'])
        .reset_index()
    )
    demand_stats['cv'] = demand_stats['std'] / demand_stats['mean']

    # Assign z-score based on volatility
    def select_z(cv):
        if cv < 0.5:
            return 1.65  # 95% service
        elif cv < 1.0:
            return 2.0   # 97.5% service
        else:
            return 2.33  # 99% service

    demand_stats['z_score'] = demand_stats['cv'].apply(select_z)

    history_quality = history_quality.merge(
        demand_stats[['location_id', 'variation_id', 'cv', 'z_score']],
        on=['location_id', 'variation_id'],
        how='left'
    )

    grouped_sales = (
        recent_12m.groupby(['date', 'location_id', 'variation_id'])
        .agg({'y': 'sum'})   # or use {'quantity_purchased': 'sum'} and rename to y
        .reset_index()
    )

    grouped_sales = grouped_sales.merge(
        history_quality[['location_id', 'variation_id', 'enough_history']],
        on=['location_id', 'variation_id'],
        how='left'
    )

    lead_time_days = 7
    results = []
    min_sigma = 1
    # prophet_forecasts dict is for testing purposes
    prophet_forecasts = {}

    # We loop over each group (one group = one item-location pair).
    # we use .iloc[0] to get the flag for that whole group.
    # If that flag is True, we proceed with Prophet on all the dates in that group.
    # If False, then we use a fallback.

    for (loc, var), group in grouped_sales.groupby(['location_id', 'variation_id']):
        enough = group['enough_history'].iloc[0]
        group = group.sort_values('date')
        prophet_df = group[['date', 'y']].rename(columns={'date': 'ds', 'y': 'y'})
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        
        # Look up z-score (defaults to 1.65 if not found) (higher z = higher safety score)
        z_row = history_quality.query('location_id == @loc and variation_id == @var')
        z = z_row['z_score'].iloc[0] if not z_row.empty else 1.65

        reorder_level = None
        replenish_level = None

        if enough:
            # Prophet fit
            try:
                m = Prophet(daily_seasonality=True)
                m.fit(prophet_df)
                future = m.make_future_dataframe(periods=lead_time_days)
                forecast = m.predict(future)
                # Save actual and forecast in a dict for plotting adn testing
                prophet_forecasts[(loc, var)] = {
                    'actual': prophet_df,  # must have ds, y
                    'forecast': forecast   # Prophet output: has ds, yhat
                }
            
                lead_forecast = forecast.tail(lead_time_days)
                demand_lt = lead_forecast['yhat'].sum()
                # Estimate the standard deviation of demand over lead time, 
                # based on Prophet’s upper/lower confidence intervals (3.29 = width for 99.9% interval; adjust if want 95%, etc. Higher number = more holding stock)
                # direct application of standard normal theory and the properties of prediction intervals 
                # reference : https://en.wikipedia.org/wiki/Prediction_interval#Normal_distribution
                sigma_lt = (lead_forecast['yhat_upper'].sum() - lead_forecast['yhat_lower'].sum()) / 3.29
                # this is buffer or safety stock
                safety_stock = z * sigma_lt
                # Set the reorder point to be demand over lead time + safety stock.
                reorder_level = int(np.round(demand_lt + safety_stock))
                # replenish = reorder + another round of lead time
                replenish_level = int(np.round(reorder_level + demand_lt))
            except Exception as e:
                last_week = prophet_df.sort_values('ds').tail(7)
                avg_daily = last_week['y'].mean() if len(last_week) else 1
                demand_lt = avg_daily * lead_time_days
                reorder_level = int(np.round(demand_lt))
                replenish_level = int(np.round(demand_lt * 2))
        else:
            sigma_lt = min_sigma
            last_week = prophet_df.sort_values('ds').tail(7)
            avg_daily = last_week['y'].mean() if len(last_week) else 1
            demand_lt = avg_daily * lead_time_days
            reorder_level = int(np.round(demand_lt))
            replenish_level = int(np.round(demand_lt * 2))

        results.append({
            'location_id': loc,
            'variation_id': var,
            'reorder_level': reorder_level,
            'replenish_level': replenish_level,
            'enough_history': enough,
            'z_score': z,
            'demand_lt': demand_lt,
            'sigma_lt': sigma_lt
        })

    results_df = pd.DataFrame(results)
    # results_df = pd.DataFrame(results)
    # Optionally save results:
    if output_path:
        results_df.to_csv(output_path, index=False)
    write_results_to_db(results_df, engine)
    return results_df


def write_results_to_db(results_df, engine):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS forecast_results (
        id INT AUTO_INCREMENT PRIMARY KEY,
        location_id INT,
        variation_id INT,
        reorder_level INT,
        replenish_level INT,
        enough_history BOOLEAN,
        z_score FLOAT,
        demand_lt FLOAT,
        sigma_lt FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    with engine.connect() as conn:
        conn.execute(sqlalchemy.text(create_table_query))
    results_df.to_sql(
        'forecast_results',
        engine,
        if_exists='append',
        index=False,
        method='multi'
    )

EXCLUDE_DBS = [
    'phpmyadmin', 'phpmyadmin2', 'horde', 'phppoint_forums', 'staging_site',
    'roundcube', 'pos', 'bntennis_site', 'mysql', 'information_schema', 'performance_schema'
]

DB_SERVERS = [
    {'host': 'database-2.ccv8sgeuslw7.us-east-1.rds.amazonaws.com', 'user': 'admin', 'password': 'GUNGBUILDYEp_69'}
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

def main():
    db_user = "admin"
    db_password = "GUNGBUILDYEp_69"
    db_host = "database-2.ccv8sgeuslw7.us-east-1.rds.amazonaws.com"
    db_port = 3306

    dbs_to_process = get_databases_to_process()

    for db_name in dbs_to_process:
        conn_str = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        print(f"Processing forecasts for DB: {db_name}")
        try:
            run_forecast_for_database(conn_str, output_path=f"forecast_{db_name}.csv")
            print(f"Finished {db_name}")
        except Exception as e:
            print(f"Failed for {db_name}: {e}")

if __name__ == "__main__":
    main()