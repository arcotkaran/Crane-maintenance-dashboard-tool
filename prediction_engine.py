# SCRIPT NAME: prediction_engine.py
# DESCRIPTION: Logic for calculating usage and predicting service dates for cranes and spreaders.
# REFACTOR: Centralized path and logger configuration.

import sqlite3
import pandas as pd
import os
import database
from datetime import datetime, timedelta
from config import Paths, Settings, logger # Import from the new centralized config

def load_service_config():
    """
    Loads the service configuration from a CSV file into a pandas DataFrame.
    """
    logger.debug(f"Attempting to load service config from: {Paths.SERVICE_CONFIG_FILE}")
    if not os.path.exists(Paths.SERVICE_CONFIG_FILE):
        logger.error(f"Service config file not found at: {Paths.SERVICE_CONFIG_FILE}")
        return None
    
    try:
        config_df = pd.read_csv(
            Paths.SERVICE_CONFIG_FILE,
            dtype=str,
            engine='python',
            keep_default_na=False
        ).set_index('task_id')
        logger.info("Service config loaded successfully.")
        return config_df
    except Exception as e:
        logger.critical(f"Failed to load or parse service_config.csv: {e}")
        return None


def get_full_history_for_metric(crane_number, tag_name):
    """
    Retrieves the entire time-series history for a specific metric for a crane.
    """
    logger.debug(f"Fetching full history for metric. Crane: {crane_number}, Tag: {tag_name}")
    if not tag_name or pd.isna(tag_name):
        logger.warning("No tag_name provided to get_full_history_for_metric. Returning empty DataFrame.")
        return pd.DataFrame()
    try:
        with sqlite3.connect(Paths.DATABASE_PATH) as conn:
            df = pd.read_sql_query(
                f"SELECT timestamp, tag_value FROM {Settings.STATS_TABLE_NAME} WHERE crane_number = ? AND tag_name = ? ORDER BY timestamp ASC",
                conn, params=(crane_number, tag_name)
            )
            if not df.empty:
                logger.debug(f"Found {len(df)} historical records.")
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                logger.debug("No historical records found.")
            return df
    except Exception as e:
        logger.error(f"DB Error on get_full_history_for_metric for {crane_number}/{tag_name}: {e}")
        return pd.DataFrame()

def get_spreader_usage_history(spreader_id, usage_tag_name):
    """
    Aggregates the total usage for a specific metric on a spreader as it moves between cranes.
    Handles new alphanumeric spreader IDs.
    """
    logger.info(f"Building usage history for Spreader ID: {spreader_id}, Metric: {usage_tag_name}")
    try:
        # FIX: Ensure spreader_id is treated as an integer for the database query.
        numeric_id = int(spreader_id)

        with sqlite3.connect(Paths.DATABASE_PATH) as conn:
            location_query = f"""
                SELECT timestamp, crane_number
                FROM {Settings.STATS_TABLE_NAME}
                WHERE tag_name = 'Spreader ID Number' AND tag_value = ?
                ORDER BY timestamp ASC
            """
            # Use the parsed numeric_id for the query
            spreader_locations_df = pd.read_sql_query(location_query, conn, params=(numeric_id,))

            if spreader_locations_df.empty:
                logger.warning(f"No location history found for spreader {spreader_id} (numeric: {numeric_id}).")
                return pd.DataFrame()

            spreader_locations_df['timestamp'] = pd.to_datetime(spreader_locations_df['timestamp'])
            spreader_locations_df['crane_change'] = spreader_locations_df['crane_number'].ne(spreader_locations_df['crane_number'].shift())
            spreader_locations_df['period_id'] = spreader_locations_df['crane_change'].cumsum()

            periods = spreader_locations_df.groupby('period_id').agg(
                crane_number=('crane_number', 'first'),
                start_time=('timestamp', 'min'),
                end_time=('timestamp', 'max')
            ).reset_index()
            logger.debug(f"Identified {len(periods)} periods of spreader attachment across different cranes.")

            all_usage_segments = []
            unique_cranes_used = periods['crane_number'].unique().tolist()
            
            placeholders = ','.join('?' for _ in unique_cranes_used)
            usage_query = f"""
                SELECT timestamp, tag_value, crane_number
                FROM {Settings.STATS_TABLE_NAME}
                WHERE tag_name = ? AND crane_number IN ({placeholders})
                ORDER BY timestamp ASC
            """
            params = [usage_tag_name] + unique_cranes_used
            full_usage_df = pd.read_sql_query(usage_query, conn, params=params)
            
            if full_usage_df.empty:
                logger.warning(f"No usage data found for metric '{usage_tag_name}' on any associated cranes.")
                return pd.DataFrame()
            
            full_usage_df['timestamp'] = pd.to_datetime(full_usage_df['timestamp'])

            for period in periods.itertuples():
                segment_df = full_usage_df[
                    (full_usage_df['crane_number'] == period.crane_number) &
                    (full_usage_df['timestamp'] >= period.start_time) &
                    (full_usage_df['timestamp'] <= period.end_time)
                ]
                if not segment_df.empty:
                    all_usage_segments.append(segment_df[['timestamp', 'tag_value']])

            if not all_usage_segments:
                logger.warning(f"No usage segments found for spreader {spreader_id} with metric {usage_tag_name}.")
                return pd.DataFrame()

            final_df = pd.concat(all_usage_segments).sort_values(by='timestamp').reset_index(drop=True)
            logger.info(f"Successfully consolidated {len(final_df)} usage records for spreader {spreader_id}.")
            return final_df

    except (ValueError, TypeError) as e:
        logger.error(f"Could not parse numeric ID from spreader ID '{spreader_id}': {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"DB Error on get_spreader_usage_history for {spreader_id}/{usage_tag_name}: {e}", exc_info=True)
        return pd.DataFrame()


def calculate_average_daily_usage(full_history_df):
    """
    Calculates a stable, long-term average daily usage from a full history DataFrame.
    """
    logger.debug(f"Calculating average daily usage from {len(full_history_df)} data points.")
    if len(full_history_df) < 2:
        logger.debug("Not enough data points (< 2) to calculate usage. Returning 0.0.")
        return 0.0
    
    df = full_history_df.copy()
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    df['value_diff'] = df['tag_value'].diff()
    df['time_diff_days'] = df['timestamp'].diff().dt.total_seconds() / (24 * 3600)
    
    df_filtered = df[df['time_diff_days'] > 0].copy()
    if df_filtered.empty:
        logger.warning("No time difference between data points. Cannot calculate daily rate.")
        return 0.0
        
    df_filtered['daily_rate'] = df_filtered['value_diff'] / df_filtered['time_diff_days']
    positive_rates = df_filtered[df_filtered['daily_rate'] >= 0]['daily_rate']
    
    avg_usage = positive_rates.mean() if not positive_rates.empty else 0.0
    logger.debug(f"Calculated average daily usage: {avg_usage}")
    return avg_usage

def predict_service_date(entity_id, entity_type, task_id, custom_limit=None):
    """
    Predicts the next service date for a given task on a specific entity (crane or spreader).
    """
    logger.info(f"Starting service date prediction for {entity_type.capitalize()}: {entity_id}, Task: {task_id}")
    service_config = load_service_config()
    if service_config is None or task_id not in service_config.index:
        logger.error(f"Task ID '{task_id}' not found in service config.")
        return {'error': f"Task ID '{task_id}' not found in config"}

    metric_info = service_config.loc[task_id]
    logger.debug(f"Metric info for task '{task_id}': {metric_info.to_dict()}")
    
    tag_name = metric_info['tag_name']
    unit = metric_info['unit']
    action_required = metric_info['action_required']
    duration_hours = metric_info['duration_hours']

    service_limit = None
    time_interval_days = None

    try:
        service_limit_val = metric_info['service_limit']
        if service_limit_val and str(service_limit_val).strip() != '':
            service_limit = float(service_limit_val)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert service_limit '{metric_info['service_limit']}' to a number for task '{task_id}'. Check service_config.csv.")

    try:
        time_interval_days_val = metric_info['service_interval_days']
        if time_interval_days_val and str(time_interval_days_val).strip() != '':
            time_interval_days = float(time_interval_days_val)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert service_interval_days '{metric_info['service_interval_days']}' to a number for task '{task_id}'. Check service_config.csv.")

    if custom_limit is not None:
        logger.debug(f"Applying custom service limit: {custom_limit}")
        service_limit = float(custom_limit)

    last_service_record = database.get_last_service_record(entity_id, entity_type, task_id)
    logger.debug(f"Last service record: {last_service_record}")
    
    if last_service_record:
        last_service_date = pd.to_datetime(last_service_record['service_date'])
        serviced_at_value = last_service_record.get('serviced_at_value', 0)
    else:
        last_service_date = datetime.now() - timedelta(days=365*5) 
        serviced_at_value = 0
        
    usage_predicted_date, time_predicted_date, final_predicted_date = None, None, None
    due_reason = "N/A"
    value_since_service, avg_daily_usage = 0, 0
    current_raw_value = 0

    if pd.notna(service_limit) and tag_name:
        logger.debug("Calculating usage-based prediction.")
        
        full_history_df = pd.DataFrame()
        if entity_type == 'crane':
            logger.debug(f"Entity is a crane. Fetching full history for metric.")
            full_history_df = get_full_history_for_metric(entity_id, tag_name)
        elif entity_type == 'spreader':
            logger.debug(f"Entity is a spreader. Fetching aggregated usage history.")
            full_history_df = get_spreader_usage_history(entity_id, tag_name)
        else:
            logger.error(f"Unknown entity_type: '{entity_type}'. Cannot fetch history.")

        if not full_history_df.empty:
            current_raw_value = full_history_df.iloc[-1]['tag_value']
            value_since_service = current_raw_value - serviced_at_value
            avg_daily_usage = calculate_average_daily_usage(full_history_df)
            if avg_daily_usage > 0:
                remaining_value = service_limit - value_since_service
                days_to_service_usage = remaining_value / avg_daily_usage
                usage_predicted_date = full_history_df.iloc[-1]['timestamp'] + timedelta(days=days_to_service_usage)
                logger.debug(f"Usage prediction: {usage_predicted_date.strftime('%Y-%m-%d') if usage_predicted_date else 'N/A'}")

    if pd.notna(time_interval_days):
        logger.debug("Calculating time-based prediction.")
        time_predicted_date = last_service_date + timedelta(days=time_interval_days)
        logger.debug(f"Time prediction: {time_predicted_date.strftime('%Y-%m-%d') if time_predicted_date else 'N/A'}")

    if usage_predicted_date and time_predicted_date:
        final_predicted_date, due_reason = (usage_predicted_date, "Usage") if usage_predicted_date < time_predicted_date else (time_predicted_date, "Time Interval")
    elif usage_predicted_date:
        final_predicted_date, due_reason = usage_predicted_date, "Usage"
    elif time_predicted_date:
        final_predicted_date, due_reason = time_predicted_date, "Time Interval"
    
    days_remaining = (final_predicted_date - datetime.now()).days if final_predicted_date else None
    logger.info(f"Final prediction for {entity_id}/{task_id}: Date={final_predicted_date}, Days Remaining={days_remaining}, Reason={due_reason}")

    return {
        'entity_id': entity_id, 
        'task_id': task_id, 
        'unit': unit, 
        'current_value': value_since_service,
        'service_limit': service_limit, 
        'service_interval_days': time_interval_days,
        'last_service_date': last_service_date,
        'avg_daily_usage': avg_daily_usage,
        'days_remaining': days_remaining,
        'predicted_date': final_predicted_date,
        'action_required': action_required,
        'due_reason': due_reason,
        'duration_hours': duration_hours,
        'error': None if final_predicted_date else "Prediction Unavailable"
    }
