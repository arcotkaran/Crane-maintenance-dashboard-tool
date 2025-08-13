# SCRIPT NAME: database.py
# DESCRIPTION: Manages all interactions with the SQLite database for crane statistics.
# REFACTOR: Centralized path, settings, and logger configuration.

import sqlite3
import pandas as pd
from datetime import datetime
from config import Paths, Settings, logger # Import from the new centralized config

def _run_migration_scripts(conn):
    """Runs data migration scripts for schema changes."""
    cursor = conn.cursor()
    logger.info("Checking for necessary database migrations...")

    try:
        # Check if the maintenance_windows table exists before running migrations
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (Settings.MAINTENANCE_WINDOWS_TABLE_NAME,))
        if cursor.fetchone() is None:
            logger.info(f"Table '{Settings.MAINTENANCE_WINDOWS_TABLE_NAME}' does not exist yet. It will be created. Skipping migration.")
            return

        # --- MIGRATION 1: crane_number to entity_id/entity_type ---
        cursor.execute(f"PRAGMA table_info({Settings.MAINTENANCE_WINDOWS_TABLE_NAME})")
        columns = [row[1] for row in cursor.fetchall()]
        if 'crane_number' in columns and 'entity_id' not in columns:
            logger.warning(f"Old schema detected for '{Settings.MAINTENANCE_WINDOWS_TABLE_NAME}' (crane_number). Migrating to entity_id/entity_type...")
            
            # Use a safe migration pattern for SQLite
            cursor.execute(f"ALTER TABLE {Settings.MAINTENANCE_WINDOWS_TABLE_NAME} RENAME TO temp_{Settings.MAINTENANCE_WINDOWS_TABLE_NAME}")
            
            # Recreate the table with the new schema (init_db will add the other new columns)
            cursor.execute(f'''
                CREATE TABLE {Settings.MAINTENANCE_WINDOWS_TABLE_NAME} (
                    id INTEGER PRIMARY KEY, 
                    entity_id TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    from_datetime TEXT NOT NULL, 
                    to_datetime TEXT NOT NULL
                )''')

            cursor.execute(f'''
                INSERT INTO {Settings.MAINTENANCE_WINDOWS_TABLE_NAME} (id, entity_id, entity_type, from_datetime, to_datetime)
                SELECT id, crane_number, 'crane', from_datetime, to_datetime 
                FROM temp_{Settings.MAINTENANCE_WINDOWS_TABLE_NAME}
            ''')

            cursor.execute(f"DROP TABLE temp_{Settings.MAINTENANCE_WINDOWS_TABLE_NAME}")
            conn.commit()
            logger.info(f"Migration for '{Settings.MAINTENANCE_WINDOWS_TABLE_NAME}' (entity_id/entity_type) completed successfully.")
        else:
            logger.info(f"'{Settings.MAINTENANCE_WINDOWS_TABLE_NAME}' schema is up to date regarding entity_id/entity_type.")

        # --- MIGRATION 2: Add detailed information columns ---
        cursor.execute(f"PRAGMA table_info({Settings.MAINTENANCE_WINDOWS_TABLE_NAME})")
        columns = [row[1] for row in cursor.fetchall()]
        new_columns = {
            'service_type': 'TEXT',
            'task_description': 'TEXT',
            'notes': 'TEXT'
        }
        
        migration_needed = False
        for col_name, col_type in new_columns.items():
            if col_name not in columns:
                migration_needed = True
                logger.info(f"Column '{col_name}' not found in '{Settings.MAINTENANCE_WINDOWS_TABLE_NAME}'. Adding it.")
                cursor.execute(f"ALTER TABLE {Settings.MAINTENANCE_WINDOWS_TABLE_NAME} ADD COLUMN {col_name} {col_type}")
        
        if migration_needed:
            conn.commit()
            logger.info("Successfully added new columns to 'maintenance_windows'.")
        else:
            logger.info("Schema for 'maintenance_windows' is already up to date with detailed columns.")

    except sqlite3.Error as e:
        logger.error(f"An error occurred during migration check for '{Settings.MAINTENANCE_WINDOWS_TABLE_NAME}': {e}")
        conn.rollback()
        raise

def init_db():
    """Initializes the database, runs migrations, and creates tables."""
    logger.info("Initializing database...")
    try:
        with sqlite3.connect(Paths.DATABASE_PATH) as conn:
            # Run migration scripts first to handle any schema updates
            _run_migration_scripts(conn)

            cursor = conn.cursor()
            
            # Create tables with the most up-to-date schema
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {Settings.STATS_TABLE_NAME} (
                    id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, crane_number TEXT NOT NULL,
                    tag_name TEXT NOT NULL, tag_value REAL, UNIQUE(crane_number, tag_name, timestamp)
                )''')
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {Settings.SERVICE_LOG_TABLE_NAME} (
                    id INTEGER PRIMARY KEY, 
                    entity_id TEXT NOT NULL, 
                    entity_type TEXT NOT NULL, 
                    task_id TEXT NOT NULL,
                    service_date TEXT NOT NULL, 
                    serviced_at_value REAL, 
                    serviced_by TEXT,
                    duration_hours REAL
                )''')
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {Settings.MAINTENANCE_WINDOWS_TABLE_NAME} (
                    id INTEGER PRIMARY KEY, 
                    entity_id TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    from_datetime TEXT NOT NULL, 
                    to_datetime TEXT NOT NULL,
                    service_type TEXT,
                    task_description TEXT,
                    notes TEXT
                )''')
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {Settings.PREDICTIONS_TABLE_NAME} (
                    entity_id TEXT,
                    task_id TEXT,
                    unit TEXT,
                    current_value REAL,
                    service_limit REAL,
                    service_interval_days REAL,
                    last_service_date TEXT,
                    avg_daily_usage REAL,
                    days_remaining INTEGER,
                    predicted_date TEXT,
                    action_required TEXT,
                    due_reason TEXT,
                    duration_hours REAL,
                    error TEXT,
                    entity_type TEXT,
                    entity_display_name TEXT,
                    PRIMARY KEY (entity_id, task_id)
                )''')
            conn.commit()
            logger.info("Database tables checked/created successfully.")
    except sqlite3.Error as e:
        logger.critical(f"Database error during initialization: {e}")
        raise

def insert_stat(timestamp, crane_number, tag_name, result):
    """Inserts a single statistic record into the database."""
    try:
        numeric_result = float(result)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert result to float. Data: {result}. Skipping insertion.")
        return
    timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')
    try:
        with sqlite3.connect(Paths.DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                INSERT INTO {Settings.STATS_TABLE_NAME} (timestamp, crane_number, tag_name, tag_value)
                VALUES (?, ?, ?, ?)
            ''', (timestamp_str, crane_number, tag_name, numeric_result))
            conn.commit()
    except sqlite3.IntegrityError:
        logger.debug(f"IntegrityError: Duplicate stat record skipped for {crane_number}, {tag_name}, {timestamp_str}")
        pass
    except sqlite3.Error as e:
        logger.error(f"Database error during stat insertion: {e} | Data: {crane_number}, {tag_name}, {numeric_result}")

def log_service_completed(entity_id, entity_type, task_id, service_date, serviced_at_value, user, duration_hours):
    """Inserts a new record into the service log table using a generic entity."""
    logger.info("Attempting to log completed service to database.")
    logger.debug(f"Data received: EntityID={entity_id}, EntityType={entity_type}, Task={task_id}, Date={service_date}, Value={serviced_at_value}, User={user}, Duration={duration_hours}")
    
    service_date_str = service_date.strftime('%Y-%m-%d %H:%M:%S')
    try:
        with sqlite3.connect(Paths.DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                INSERT INTO {Settings.SERVICE_LOG_TABLE_NAME} 
                (entity_id, entity_type, task_id, service_date, serviced_at_value, serviced_by, duration_hours)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (entity_id, entity_type, task_id, service_date_str, serviced_at_value, user, duration_hours))
            conn.commit()
        logger.info(f"Successfully inserted service log for Entity {entity_id} ({entity_type}), Task {task_id}.")
        return True
    except sqlite3.Error as e:
        logger.error(f"DB Error on log_service_completed: {e}")
        return False

def get_last_service_record(entity_id, entity_type, task_id):
    """Retrieves the most recent service log entry for a specific entity and task_id."""
    logger.debug(f"Fetching last service record for Entity {entity_id} ({entity_type}), Task {task_id}.")
    try:
        with sqlite3.connect(Paths.DATABASE_PATH) as conn:
            query = f"""
                SELECT id, service_date, serviced_at_value 
                FROM {Settings.SERVICE_LOG_TABLE_NAME} 
                WHERE entity_id = ? AND entity_type = ? AND task_id = ? 
                ORDER BY service_date DESC, serviced_at_value DESC 
                LIMIT 1
            """
            df = pd.read_sql_query(query, conn, params=(entity_id, entity_type, task_id))
            if not df.empty:
                logger.debug("Found last service record.")
                return df.to_dict('records')[0]
            else:
                logger.debug("No previous service record found.")
                return None
    except Exception as e:
        logger.error(f"DB Error on get_last_service_record for {entity_id}/{entity_type}/{task_id}: {e}")
        return None

def get_all_service_logs_for_task(entity_id, entity_type, task_id):
    """Retrieves the COMPLETE service history for a specific entity and task_id."""
    logger.debug(f"Fetching all service logs for Entity {entity_id} ({entity_type}), Task {task_id}.")
    try:
        with sqlite3.connect(Paths.DATABASE_PATH) as conn:
            query = f"""
                SELECT id, service_date, serviced_at_value, serviced_by 
                FROM {Settings.SERVICE_LOG_TABLE_NAME} 
                WHERE entity_id = ? AND entity_type = ? AND task_id = ? 
                ORDER BY service_date DESC, serviced_at_value DESC
            """
            return pd.read_sql_query(query, conn, params=(entity_id, entity_type, task_id))
    except Exception as e:
        logger.error(f"DB Error on get_all_service_logs_for_task for {entity_id}/{entity_type}/{task_id}: {e}")
        return pd.DataFrame()

def get_all_service_logs():
    """Retrieves all service log entries from the database."""
    logger.debug("Fetching all service logs from database.")
    try:
        with sqlite3.connect(Paths.DATABASE_PATH) as conn:
            return pd.read_sql_query(
                f"SELECT * FROM {Settings.SERVICE_LOG_TABLE_NAME} ORDER BY service_date DESC",
                conn
            )
    except Exception as e:
        logger.error(f"DB Error on get_all_service_logs: {e}")
        return pd.DataFrame()

def delete_service_log(log_id):
    """Deletes a specific service log entry by its ID."""
    logger.info(f"Attempting to delete service log with ID: {log_id}")
    try:
        with sqlite3.connect(Paths.DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {Settings.SERVICE_LOG_TABLE_NAME} WHERE id = ?", (log_id,))
            conn.commit()
            logger.info(f"Successfully deleted service log ID: {log_id}")
        return True
    except sqlite3.Error as e:
        logger.error(f"DB Error on delete_service_log for ID {log_id}: {e}")
        return False

def add_maintenance_window(entity_id, entity_type, from_dt, to_dt, service_type=None, task_description=None, notes=None):
    """
    Adds a new maintenance window to the database for a generic entity.
    Accepts optional details for CSV imports.
    """
    logger.info(f"Attempting to add maintenance window for {entity_type} {entity_id} from {from_dt} to {to_dt}")
    from_dt_str = from_dt.strftime('%Y-%m-%d %H:%M:%S')
    to_dt_str = to_dt.strftime('%Y-%m-%d %H:%M:%S')
    try:
        with sqlite3.connect(Paths.DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                INSERT INTO {Settings.MAINTENANCE_WINDOWS_TABLE_NAME} 
                (entity_id, entity_type, from_datetime, to_datetime, service_type, task_description, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (str(entity_id), entity_type, from_dt_str, to_dt_str, service_type, task_description, notes))
            conn.commit()
            logger.info("Successfully added maintenance window.")
        return True
    except sqlite3.Error as e:
        logger.error(f"DB Error on add_maintenance_window: {e}")
        return False

def check_maintenance_window_exists(entity_id, from_dt, to_dt):
    """Checks if a maintenance window with the same entity and times already exists."""
    logger.debug(f"Checking for existing window for Entity {entity_id} from {from_dt} to {to_dt}")
    # Format datetimes to strings to match the database TEXT storage format
    from_dt_str = from_dt.strftime('%Y-%m-%d %H:%M:%S')
    to_dt_str = to_dt.strftime('%Y-%m-%d %H:%M:%S')
    try:
        with sqlite3.connect(Paths.DATABASE_PATH) as conn:
            query = f"""
                SELECT id FROM {Settings.MAINTENANCE_WINDOWS_TABLE_NAME}
                WHERE entity_id = ? AND from_datetime = ? AND to_datetime = ?
                LIMIT 1
            """
            cursor = conn.cursor()
            cursor.execute(query, (str(entity_id), from_dt_str, to_dt_str))
            if cursor.fetchone():
                logger.debug("Found existing maintenance window.")
                return True
            else:
                logger.debug("No existing maintenance window found.")
                return False
    except Exception as e:
        logger.error(f"DB Error on check_maintenance_window_exists for {entity_id}: {e}", exc_info=True)
        return False # Fail safe, assume it doesn't exist to avoid blocking valid imports

def check_for_conflicting_maintenance_windows(entity_id, from_dt, to_dt):
    """
    Checks if a new maintenance window conflicts with any existing windows for the same entity.
    A conflict occurs if the time ranges overlap.
    """
    logger.debug(f"Checking for conflicting windows for Entity {entity_id} from {from_dt} to {to_dt}")
    from_dt_str = from_dt.strftime('%Y-%m-%d %H:%M:%S')
    to_dt_str = to_dt.strftime('%Y-%m-%d %H:%M:%S')

    try:
        with sqlite3.connect(Paths.DATABASE_PATH) as conn:
            # The logic for overlap is: (StartA < EndB) and (StartB < EndA)
            query = f"""
                SELECT id, from_datetime, to_datetime
                FROM {Settings.MAINTENANCE_WINDOWS_TABLE_NAME}
                WHERE
                    entity_id = ? AND
                    ? < to_datetime AND
                    from_datetime < ?
                LIMIT 1
            """
            cursor = conn.cursor()
            cursor.execute(query, (str(entity_id), from_dt_str, to_dt_str))
            conflict = cursor.fetchone()
            if conflict:
                logger.warning(f"Found conflicting maintenance window. New: {from_dt_str}-{to_dt_str}, Existing (ID {conflict[0]}): {conflict[1]}-{conflict[2]}")
                return True
            else:
                logger.debug("No conflicting windows found.")
                return False
    except Exception as e:
        logger.error(f"DB Error on check_for_conflicting_maintenance_windows for {entity_id}: {e}", exc_info=True)
        return False # Fail safe

def get_all_maintenance_windows():
    """Retrieves all maintenance windows from the database."""
    logger.debug("Fetching all maintenance windows from database.")
    try:
        with sqlite3.connect(Paths.DATABASE_PATH) as conn:
            df = pd.read_sql_query(
                f"SELECT * FROM {Settings.MAINTENANCE_WINDOWS_TABLE_NAME} ORDER BY from_datetime DESC",
                conn
            )
            if not df.empty:
                df['from_datetime'] = pd.to_datetime(df['from_datetime'])
                df['to_datetime'] = pd.to_datetime(df['to_datetime'])
            return df
    except Exception as e:
        if "no such column: from_datetime" in str(e):
             logger.warning("get_all_maintenance_windows failed, possibly due to migration. Returning empty DataFrame.")
             return pd.DataFrame()
        logger.error(f"DB Error on get_all_maintenance_windows: {e}")
        return pd.DataFrame()

def delete_maintenance_window(window_id):
    """Deletes a specific maintenance window by its ID."""
    logger.info(f"Attempting to delete maintenance window with ID: {window_id}")
    try:
        with sqlite3.connect(Paths.DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {Settings.MAINTENANCE_WINDOWS_TABLE_NAME} WHERE id = ?", (window_id,))
            conn.commit()
            logger.info(f"Successfully deleted maintenance window ID: {window_id}")
        return True
    except sqlite3.Error as e:
        logger.error(f"DB Error on delete_maintenance_window for ID {window_id}: {e}")
        return False

def get_spreader_location_history(spreader_id: str):
    """
    Retrieves and processes the location history of a specific spreader.
    Handles new alphanumeric spreader IDs by parsing the numeric part.
    """
    logger.info(f"Fetching location history for Spreader ID: {spreader_id}")
    try:
        # Extract the numeric part of the ID for the database query.
        # e.g., 'SP001' -> 1
        if isinstance(spreader_id, str) and spreader_id.startswith('SP'):
            numeric_id = int(spreader_id[2:])
        else:
            numeric_id = int(spreader_id) # Fallback for old numeric IDs or other formats

        with sqlite3.connect(Paths.DATABASE_PATH) as conn:
            query = f"""
                SELECT timestamp, crane_number
                FROM {Settings.STATS_TABLE_NAME}
                WHERE tag_name = 'Spreader ID Number' AND tag_value = ?
                ORDER BY timestamp ASC
            """
            # Use the parsed numeric_id in the query
            df = pd.read_sql_query(query, conn, params=(numeric_id,))

            if df.empty:
                logger.warning(f"No location history found for spreader {spreader_id} (numeric: {numeric_id}).")
                return pd.DataFrame()

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['block'] = (df['crane_number'] != df['crane_number'].shift()).cumsum()
            periods = df.groupby('block').agg(
                crane_number=('crane_number', 'first'),
                from_date=('timestamp', 'min')
            ).reset_index(drop=True)
            periods['to_date'] = periods['from_date'].shift(-1)

            if not periods.empty:
                # Ensure the last period extends to the most recent timestamp
                periods.loc[periods.index[-1], 'to_date'] = df['timestamp'].max()
            
            periods.dropna(subset=['to_date'], inplace=True)
            # Filter out zero-duration events which can occur if timestamps are identical
            periods = periods[periods['from_date'] != periods['to_date']]

            if periods.empty:
                return pd.DataFrame()

            periods['duration_days'] = (periods['to_date'] - periods['from_date']).dt.total_seconds() / (24 * 3600)
            history_df = periods.sort_values(by='from_date', ascending=False).reset_index(drop=True)
            
            logger.info(f"Successfully processed location history for spreader {spreader_id}.")
            return history_df[['crane_number', 'from_date', 'to_date', 'duration_days']]

    except (ValueError, TypeError) as e:
        logger.error(f"Could not parse numeric ID from spreader ID '{spreader_id}': {e}", exc_info=True)
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error getting spreader location history for ID {spreader_id}: {e}", exc_info=True)
        return pd.DataFrame()


def get_entities_with_new_data(since_timestamp: str):
    """
    Identifies cranes and spreaders that have new statistical data since the given timestamp.

    Args:
        since_timestamp (str): An ISO format timestamp string.

    Returns:
        dict: A dictionary with two keys, 'cranes' and 'spreaders', containing lists
              of their respective IDs that need updating.
    """
    logger.info(f"Checking for entities with new data since {since_timestamp}")
    entities_to_update = {'cranes': set(), 'spreaders': set()}

    try:
        with sqlite3.connect(Paths.DATABASE_PATH) as conn:
            # 1. Find all cranes with new raw data since the last check
            crane_query = f"""
                SELECT DISTINCT crane_number
                FROM {Settings.STATS_TABLE_NAME}
                WHERE timestamp > ?
            """
            new_data_cranes_df = pd.read_sql_query(crane_query, conn, params=(since_timestamp,))

            if new_data_cranes_df.empty:
                logger.info("No new stat entries found. No updates needed.")
                return {'cranes': [], 'spreaders': []}

            cranes_with_new_data = set(new_data_cranes_df['crane_number'].unique())
            entities_to_update['cranes'].update(cranes_with_new_data)
            logger.debug(f"Found {len(cranes_with_new_data)} cranes with new raw stats: {cranes_with_new_data}")

            # 2. For those cranes, find out which spreaders are currently attached to them
            if not cranes_with_new_data:
                return {'cranes': [], 'spreaders': []}

            placeholders = ','.join('?' for _ in cranes_with_new_data)
            spreader_query = f"""
                WITH LatestSpreader AS (
                    SELECT
                        crane_number,
                        tag_value,
                        ROW_NUMBER() OVER (PARTITION BY crane_number ORDER BY timestamp DESC) as rn
                    FROM {Settings.STATS_TABLE_NAME}
                    WHERE tag_name = 'Spreader ID Number' AND crane_number IN ({placeholders})
                )
                SELECT DISTINCT tag_value
                FROM LatestSpreader
                WHERE rn = 1 AND tag_value IS NOT NULL
            """
            params = list(cranes_with_new_data)
            affected_spreaders_df = pd.read_sql_query(spreader_query, conn, params=params)

            if not affected_spreaders_df.empty:
                affected_spreader_ids = set(affected_spreaders_df['tag_value'].astype(int).unique())
                entities_to_update['spreaders'].update(affected_spreader_ids)
                logger.debug(f"Found {len(affected_spreader_ids)} associated spreaders to update: {affected_spreader_ids}")

    except Exception as e:
        logger.error(f"DB Error while checking for new entity data: {e}", exc_info=True)
        return {'cranes': [], 'spreaders': []}

    # Convert sets to lists before returning
    final_updates = {
        'cranes': list(entities_to_update['cranes']),
        'spreaders': list(entities_to_update['spreaders'])
    }
    logger.info(f"Entities requiring an update: {final_updates}")
    return final_updates

def store_predictions(preds_df):
    """Stores the prediction results DataFrame in the database, replacing old data."""
    if preds_df.empty:
        logger.warning("Attempted to store an empty predictions DataFrame. Aborting.")
        return

    try:
        # Set the index to ensure the primary key is correctly maintained on replace.
        preds_df_indexed = preds_df.set_index(['entity_id', 'task_id'])
        with sqlite3.connect(Paths.DATABASE_PATH) as conn:
            preds_df_indexed.to_sql(
                Settings.PREDICTIONS_TABLE_NAME,
                conn,
                if_exists='replace',
                index=True
            )
            logger.info(f"Successfully stored {len(preds_df)} predictions in the database.")
    except Exception as e:
        logger.error(f"DB Error on store_predictions: {e}", exc_info=True)

def get_all_predictions():
    """Retrieves all cached prediction results from the database."""
    logger.info("Attempting to load predictions from the database cache.")
    try:
        with sqlite3.connect(Paths.DATABASE_PATH) as conn:
            # Explicitly parse date columns to prevent errors.
            df = pd.read_sql_query(
                f"SELECT * FROM {Settings.PREDICTIONS_TABLE_NAME}",
                conn,
                parse_dates=['last_service_date', 'predicted_date']
            )
            if not df.empty:
                logger.info(f"Successfully loaded {len(df)} predictions from cache.")
                return df
            else:
                logger.warning("Prediction cache is empty.")
                return pd.DataFrame()
    except Exception as e:
        logger.error(f"Could not read predictions from cache, it may not exist yet. Error: {e}")
        return pd.DataFrame()