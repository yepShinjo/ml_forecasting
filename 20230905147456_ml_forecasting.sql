-- ml_forecasting --
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

ALTER TABLE phppos_location_item_variations
    ADD COLUMN IF NOT EXISTS forecasted_reorder_level INT DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS forecasted_replenish_level INT DEFAULT NULL;

ALTER TABLE phppos_location_items
    ADD COLUMN IF NOT EXISTS forecasted_reorder_level INT DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS forecasted_replenish_level INT DEFAULT NULL;
