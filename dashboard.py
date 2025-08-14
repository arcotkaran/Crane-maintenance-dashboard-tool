import streamlit as st
import bcrypt
import pandas as pd
import prediction_engine
import database
import auth
from datetime import datetime, timedelta
import altair as alt
import os
import sqlite3
from config import Paths, Settings, logger # Import from the new centralized config
from datetime import datetime, timezone
import numpy as np


# --- Global Configuration ---
st.set_page_config(page_title="Crane & Spreader Maintenance Dashboard", page_icon="🏗️", layout="wide")

# --- Dynamic CSS for Light/Dark Mode ---
# This CSS block now defines styles for both themes. Streamlit will apply the correct one.
st.markdown("""
<style>
/* --- Light Mode Styles --- */
body[data-theme="light"] [data-testid="stMetric"], 
body[data-theme="light"] .spreader-card, 
body[data-theme="light"] .health-widget-container {
    background-color: #F0F2F6;
    border: 1px solid #DDDDDD;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    color: #333333;
}
body[data-theme="light"] [data-testid="stMetricLabel"] { color: #555555; }
body[data-theme="light"] div[role="radiogroup"] > label {
    background-color: #F0F2F6;
    border: 1px solid #DDDDDD;
}
body[data-theme="light"] div[role="radiogroup"] > label:hover {
    background-color: #E0E0E0;
    box-shadow: 0 0 10px #007BFF;
}
body[data-theme="light"] div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) {
    background-color: #007BFF;
    color: #FFFFFF;
    border: 1px solid #007BFF;
}
body[data-theme="light"] .inner-ring { background-color: #F0F2F6; }

/* --- Dark Mode Styles --- */
body[data-theme="dark"] [data-testid="stMetric"], 
body[data-theme="dark"] .spreader-card, 
body[data-theme="dark"] .health-widget-container {
    background-color: #1A1A1A;
    border: 1px solid #333333;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
    color: #EAEAEA;
}
body[data-theme="dark"] [data-testid="stMetricLabel"] { color: #888888; }
body[data-theme="dark"] div[role="radiogroup"] > label {
    background-color: #1A1A1A;
    border: 1px solid #333333;
}
body[data-theme="dark"] div[role="radiogroup"] > label:hover {
    background-color: #2a2a2a;
    box-shadow: 0 0 10px #00AFFF;
}
body[data-theme="dark"] div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) {
    background-color: #00AFFF;
    color: #000000;
    font-weight: bold;
    border: 1px solid #00AFFF;
}
body[data-theme="dark"] .inner-ring { background-color: #1A1A1A; }

/* --- Common Component Styles --- */
[data-testid="stMetric"], .spreader-card, .health-widget-container {
    border-radius: 12px;
    padding: 20px;
}
[data-testid="stMetricLabel"] {
    font-size: 1.1rem;
    font-weight: 500;
}
div[role="radiogroup"] > label {
    display: inline-block;
    padding: 10px 16px;
    margin: 0 4px;
    border-radius: 8px;
    cursor: pointer;
    transition: background-color 0.3s, color 0.3s, box-shadow 0.3s;
}
.health-widget-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    height: 100%;
    min-height: 250px;
}
.rings-container {
    position: relative;
    width: 150px;
    height: 150px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 15px;
    margin-bottom: 20px;
}
.health-ring {
    border-radius: 50%;
    position: absolute;
    display: grid;
    place-items: center;
}
.outer-ring {
    width: 150px;
    height: 150px;
    padding: 15px;
    background-clip: content-box;
}
.inner-ring {
    width: 100px;
    height: 100px;
    padding: 15px;
    background-clip: content-box;
}
.health-status {
    font-size: 1.2rem;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_fleet_name_maps():
    """
    Reads the fleet map file and returns a comprehensive set of mapping dictionaries
    for all entity types (cranes and spreaders).
    """
    if not os.path.exists(Paths.FLEET_MAP_FILE):
        st.error(f"Critical Error: Fleet map file not found at {Paths.FLEET_MAP_FILE}")
        logger.error(f"Critical Error: Fleet map file not found at {Paths.FLEET_MAP_FILE}")
        # Return empty dicts to prevent crashes downstream
        return {}, {}, {}, {}

    try:
        df = pd.read_csv(Paths.FLEET_MAP_FILE)
        
        # --- Create Mappings ---
        # 1. internal_name -> csv_name (e.g., 'RMG01' -> 'CASC01 EAST')
        # This will be the primary map for displaying friendly names.
        internal_to_display = pd.Series(df.csv_name.values, index=df.internal_name).to_dict()

        # 2. csv_name -> internal_name (e.g., 'CASC01 EAST' -> 'RMG01')
        # Useful for converting user selections back to internal IDs for processing.
        display_to_internal = pd.Series(df.internal_name.values, index=df.csv_name).to_dict()

        # --- Spreader-Specific Mappings ---
        spreader_df = df[df['numeric_id'].notna()].copy()
        spreader_df['numeric_id'] = spreader_df['numeric_id'].astype(int)

        # 3. numeric_id -> csv_name (e.g., 29336 -> 'SP001')
        numeric_id_to_display = pd.Series(spreader_df.csv_name.values, index=spreader_df.numeric_id).to_dict()
        
        # 4. csv_name -> numeric_id (e.g., 'SP001' -> 29336)
        display_to_numeric_id = pd.Series(spreader_df.numeric_id.values, index=spreader_df.csv_name).to_dict()

        logger.info("Successfully loaded all fleet name maps.")
        return internal_to_display, display_to_internal, numeric_id_to_display, display_to_numeric_id

    except Exception as e:
        logger.error(f"Failed to load or process fleet name maps: {e}", exc_info=True)
        st.error("Failed to load fleet mapping configuration.")
        return {}, {}, {}, {}

# --- Initialize the new name maps ---
INTERNAL_TO_DISPLAY_NAME, DISPLAY_TO_INTERNAL_NAME, NUMERIC_ID_TO_DISPLAY_NAME, DISPLAY_TO_NUMERIC_ID = get_fleet_name_maps()



# --- Helper Functions for Spreaders ---
@st.cache_data
def get_spreader_name_maps():
    """
    Reads the fleet map file, filters for spreaders based on the presence of a numeric_id,
    and returns mapping dictionaries.
    """
    if not os.path.exists(Paths.FLEET_MAP_FILE):
        st.error(f"Fleet map file not found: {Paths.FLEET_MAP_FILE}")
        logger.error(f"Fleet map file not found: {Paths.FLEET_MAP_FILE}")
        return {}, {}
    try:
        df = pd.read_csv(Paths.FLEET_MAP_FILE)
        
        # Ensure required columns exist for spreader mapping
        if 'csv_name' not in df.columns or 'numeric_id' not in df.columns:
            st.error("Fleet map file is missing 'csv_name' or 'numeric_id' columns.")
            logger.error("Fleet map file is missing 'csv_name' or 'numeric_id' columns.")
            return {}, {}

        # A row represents a spreader if 'numeric_id' is not empty/NaN.
        spreader_df = df[df['numeric_id'].notna()].copy()
        
        if spreader_df.empty:
            logger.warning("No spreader entries with numeric_id found in fleet_map.csv.")
            return {}, {}

        # Ensure numeric_id is integer for correct database lookups
        spreader_df['numeric_id'] = spreader_df['numeric_id'].astype(int)

        # name_to_id should map the display name (csv_name) to the numeric_id
        name_to_id = pd.Series(spreader_df['numeric_id'].values, index=spreader_df['csv_name'].astype(str)).to_dict()
        
        # id_to_name should map the numeric_id back to the display name (csv_name)
        id_to_name = {v: k for k, v in name_to_id.items()}
        
        logger.info(f"Successfully loaded spreader name maps from fleet_map.csv. Found {len(spreader_df)} spreaders.")
        logger.debug(f"Spreader name_to_id map: {name_to_id}")
        logger.debug(f"Spreader id_to_name map: {id_to_name}")
        return name_to_id, id_to_name
    except Exception as e:
        logger.error(f"Failed to load or process spreader mappings from fleet map file: {e}", exc_info=True)
        return {}, {}


@st.cache_data(ttl=60)
def find_spreader_conflicts():
    """
    Identifies cranes reporting the same spreader ID based on their latest report.
    """
    try:
        with sqlite3.connect(Paths.DATABASE_PATH) as conn:
            query = f"""
                WITH LastReportedSpreader AS (
                    SELECT
                        crane_number,
                        tag_value AS spreader_id
                    FROM (
                        SELECT
                            crane_number,
                            tag_value,
                            ROW_NUMBER() OVER(PARTITION BY crane_number ORDER BY timestamp DESC) as rn
                        FROM {Settings.STATS_TABLE_NAME}
                        WHERE tag_name = 'Spreader ID Number'
                    )
                    WHERE rn = 1
                ),
                ConflictedSpreaderIDs AS (
                    SELECT spreader_id
                    FROM LastReportedSpreader
                    GROUP BY spreader_id
                    HAVING COUNT(crane_number) > 1
                )
                SELECT lrs.crane_number
                FROM LastReportedSpreader lrs
                JOIN ConflictedSpreaderIDs csi ON lrs.spreader_id = csi.spreader_id;
            """
            conflicted_cranes_df = pd.read_sql_query(query, conn)
            
            if not conflicted_cranes_df.empty:
                conflicted_list = conflicted_cranes_df['crane_number'].tolist()
                logger.warning(f"Persistent spreader conflicts found on cranes: {conflicted_list}")
                return conflicted_list
            return []
    except Exception as e:
        logger.error(f"Database error while finding persistent spreader conflicts: {e}")
        return []

@st.cache_data(ttl=300)
def get_all_spreader_statuses():
    """
    Gets the current status (latest location and duration) for all spreaders.
    """
    logger.info("Aggregating current status for all spreaders.")
    all_statuses = []
    spreader_name_to_id, _ = get_spreader_name_maps()
    conflicted_cranes = find_spreader_conflicts()

    for spreader_name, spreader_id in spreader_name_to_id.items():
        history_df = database.get_spreader_location_history(spreader_id)
        if not history_df.empty:
            latest_record = history_df.iloc[0]
            current_location = latest_record['crane_number']
            duration_days = (datetime.now() - latest_record['from_date']).days
            last_seen_date = latest_record['to_date']
            status = "Active"
            is_conflicted = current_location in conflicted_cranes
        else:
            current_location = 'N/A'
            duration_days = 'N/A'
            last_seen_date = 'N/A'
            status = 'No History'
            is_conflicted = False

        all_statuses.append({
            'spreader_name': spreader_name,
            'crane_number': current_location,
            'duration_days': duration_days,
            'last_seen_date': last_seen_date,
            'status': status,
            'is_conflicted': is_conflicted
        })

    master_df = pd.DataFrame(all_statuses)
    logger.info(f"Successfully aggregated statuses for {len(master_df)} spreaders.")
    return master_df

def _validate_maintenance_plan(df):
    """
    Parses and validates a maintenance plan DataFrame from a CSV file, using a centralized
    fleet map for ID translation and providing granular logging.
    """
    valid_records = []
    duplicate_records = []
    error_records = []

    # --- Load the fleet map and create a unified translator ---
    try:
        fleet_map_df = pd.read_csv(Paths.FLEET_MAP_FILE)
        if 'csv_name' not in fleet_map_df.columns or 'internal_name' not in fleet_map_df.columns:
             raise ValueError("fleet_map.csv must contain 'csv_name' and 'internal_name' columns.")

        fleet_translator = {}
        # Process cranes (where numeric_id is NaN)
        crane_df = fleet_map_df[fleet_map_df['numeric_id'].isna()]
        crane_translator = pd.Series(crane_df['internal_name'].values, index=crane_df['csv_name'].astype(str)).to_dict()
        fleet_translator.update(crane_translator)

        # Process spreaders (where numeric_id is not NaN)
        spreader_df = fleet_map_df[fleet_map_df['numeric_id'].notna()]
        # Map csv_name to the numeric_id for spreaders
        spreader_translator = pd.Series(spreader_df['numeric_id'].astype(int).values, index=spreader_df['csv_name'].astype(str)).to_dict()
        fleet_translator.update(spreader_translator)

        logger.info("Successfully built unified translator from fleet_map.csv for maintenance plan validation.")
        logger.debug(f"Unified translator map: {fleet_translator}")

    except Exception as e:
        logger.error(f"CRITICAL: Failed to read or process fleet_map.csv for validation: {e}", exc_info=True)
        return [], [], [{'errors': f"System Error: Could not process fleet_map.csv. Error: {e}"}]


    COLUMN_MAP = {
        "Fleet #": "entity_id",
        "Vehicle Type": "entity_type",
        "Scheduled Start Date": "from_datetime",
        "Scheduled End Date": "to_datetime",
        "Service Type": "service_type",
        "Service Alias": "task_description",
        "Repairer Notes": "notes"
    }

    required_csv_columns = ["Fleet #", "Scheduled Start Date", "Scheduled End Date", "Vehicle Type"]
    for col in required_csv_columns:
        if col not in df.columns:
            logger.error(f"CSV Upload Error: Missing required column '{col}'.")
            return [], [], [{'errors': f"Missing required column in CSV: '{col}'"}]

    for index, row in df.iterrows():
        record = row.to_dict()
        error_messages = []
        logger.debug(f"--- Processing Row {index+1}: {record} ---")
        
        from_datetime, to_datetime = None, None

        # --- Rule 1: Missing Data Validation ---
        entity_id_from_csv = record.get("Fleet #")
        start_date_str = record.get("Scheduled Start Date")
        end_date_str = record.get("Scheduled End Date")
        vehicle_type_raw = record.get("Vehicle Type")

        if pd.isna(entity_id_from_csv) or str(entity_id_from_csv).strip() == '':
            error_messages.append("Missing required value for 'Fleet #'.")
        if pd.isna(start_date_str) or str(start_date_str).strip() == '':
            error_messages.append("Missing required value for 'Scheduled Start Date'.")
        if pd.isna(end_date_str) or str(end_date_str).strip() == '':
            error_messages.append("Missing required value for 'Scheduled End Date'.")
        if pd.isna(vehicle_type_raw) or str(vehicle_type_raw).strip() == '':
            error_messages.append("Missing required value for 'Vehicle Type'.")

        if error_messages:
            record['errors'] = ", ".join(error_messages)
            error_records.append(record)
            logger.debug(f"Row {index+1} classified as ERROR due to missing data: {error_messages}")
            continue

        # --- Rule 2: Fleet # Translation ---
        cleaned_csv_name = str(entity_id_from_csv).strip()
        internal_id = fleet_translator.get(cleaned_csv_name)

        if not internal_id:
            error_msg = f"Error: Fleet # '{cleaned_csv_name}' not found in mapping file."
            error_messages.append(error_msg)
            record['errors'] = ", ".join(error_messages)
            # BUG FIX: Append the record with the translation error to the error list.
            error_records.append(record)
            logger.debug(f"Translation Failure for row {index+1}: '{cleaned_csv_name}' not found.")
            logger.debug(f"Row {index+1} classified as ERROR.")
            continue
        else:
            logger.debug(f"Translation Success for row {index+1}: Translated '{cleaned_csv_name}' to '{internal_id}'.")

        # --- Rule 3: Date Format and Logic Validation ---
        try:
            from_datetime = pd.to_datetime(start_date_str, dayfirst=True)
            to_datetime = pd.to_datetime(end_date_str, dayfirst=True)
            if to_datetime < from_datetime:
                error_messages.append("Date error: 'Scheduled End Date' is before 'Scheduled Start Date'.")
            else:
                logger.debug(f"Row {index+1}: Successfully parsed dates. From: {from_datetime}, To: {to_datetime}.")
        except Exception as e:
            logger.debug(f"Date parsing failed for row {index+1}: {e}")
            error_messages.append("Date error: Could not parse dates. Expected format like 'DD/MM/YYYY HH:MM'.")
        
        # --- Rule 4: Vehicle Type Validation ---
        vehicle_type_str = str(vehicle_type_raw).lower().strip()
        if 'crane' in vehicle_type_str:
            entity_type = 'crane'
        elif 'spreader' in vehicle_type_str or 'bromma' in vehicle_type_str:
            entity_type = 'spreader'
        else:
            entity_type = 'unknown'
            error_messages.append(f"Invalid 'Vehicle Type': '{vehicle_type_raw}' is not recognized.")
        logger.debug(f"Row {index+1}: Vehicle type identified as '{entity_type}'.")

        if error_messages:
            record['errors'] = ", ".join(error_messages)
            error_records.append(record)
            logger.debug(f"Row {index+1} classified as ERROR due to validation issues: {error_messages}")
            continue
        
        # --- Rule 5: Duplicate and Conflict Check against Database ---
        if from_datetime and to_datetime:
            if database.check_maintenance_window_exists(internal_id, from_datetime, to_datetime):
                duplicate_records.append(record)
                logger.debug(f"Row {index+1} classified as DUPLICATE. Exact window already exists.")
                continue
            
            if database.check_for_conflicting_maintenance_windows(internal_id, from_datetime, to_datetime):
                error_messages.append("Conflict error: The maintenance window overlaps with an existing one.")
                record['errors'] = ", ".join(error_messages)
                error_records.append(record)
                logger.debug(f"Row {index+1} classified as ERROR due to a schedule conflict.")
                continue
            logger.debug(f"Row {index+1}: Database checks passed (no exact duplicate or conflict).")

        # --- Rule 6: Valid Record Preparation ---
        valid_record = {
            COLUMN_MAP["Fleet #"]: internal_id,
            COLUMN_MAP["Vehicle Type"]: entity_type,
            COLUMN_MAP["Scheduled Start Date"]: from_datetime,
            COLUMN_MAP["Scheduled End Date"]: to_datetime,
            COLUMN_MAP["Service Type"]: record.get("Service Type"),
            COLUMN_MAP["Service Alias"]: record.get("Service Alias"),
            COLUMN_MAP["Repairer Notes"]: record.get("Repairer Notes"),
            "original_fleet_#": cleaned_csv_name
        }
        valid_records.append(valid_record)
        logger.debug(f"Row {index+1} classified as VALID.")
    
    logger.info(f"CSV validation complete. Valid: {len(valid_records)}, Duplicates: {len(duplicate_records)}, Errors: {len(error_records)}")
    
    return valid_records, duplicate_records, error_records




@st.cache_data(ttl=300)
def run_initial_predictions():
    """Runs predictions for ALL cranes and spreaders. Used for the first page load."""
    logger.info("Executing run_initial_predictions() for all cranes and spreaders.")
    # ... (The code here is IDENTICAL to your existing run_all_predictions function) ...
    config = get_config()
    if config is None:
        return pd.DataFrame()

    all_preds = []
    crane_internal_names = [internal for display, internal in DISPLAY_TO_INTERNAL_NAME.items() if 'RMG' in internal]
    crane_tasks_df = config[config['category'] != 'Spreader']
    for task_id in crane_tasks_df.index:
        for internal_name in crane_internal_names:
            pred = prediction_engine.predict_service_date(internal_name, 'crane', task_id)
            if pred: pred['entity_type'] = 'crane'
            all_preds.append(pred)

    spreader_tasks_df = config[config['category'] == 'Spreader']
    for display_name, numeric_id in DISPLAY_TO_NUMERIC_ID.items():
        for task_id in spreader_tasks_df.index:
            pred = prediction_engine.predict_service_date(numeric_id, 'spreader', task_id)
            if pred: pred['entity_type'] = 'spreader'
            all_preds.append(pred)

    preds_df = pd.DataFrame([p for p in all_preds if p is not None and not p.get('error')])
    if preds_df.empty: return pd.DataFrame()

    def map_id_to_display_name(row):
        if row['entity_type'] == 'crane': return INTERNAL_TO_DISPLAY_NAME.get(row['entity_id'], row['entity_id'])
        elif row['entity_type'] == 'spreader': return NUMERIC_ID_TO_DISPLAY_NAME.get(int(row['entity_id']), row['entity_id'])
        return row['entity_id']

    preds_df['entity_display_name'] = preds_df.apply(map_id_to_display_name, axis=1)
    if 'crane' in preds_df.columns: preds_df.drop(columns=['crane'], inplace=True)
    logger.info(f"Successfully generated {len(preds_df)} initial predictions.")
    return preds_df


def run_targeted_predictions(entities_to_update: dict):
    """Runs predictions only for the specified list of cranes and spreaders."""
    if not entities_to_update.get('cranes') and not entities_to_update.get('spreaders'):
        logger.info("run_targeted_predictions called with no entities to update.")
        return pd.DataFrame()

    logger.info(f"Executing targeted predictions for: {entities_to_update}")
    config = get_config()
    if config is None: return pd.DataFrame()
    all_preds = []

    # Predict for specific cranes that have new data
    if entities_to_update.get('cranes'):
        crane_tasks_df = config[config['category'] != 'Spreader']
        for task_id in crane_tasks_df.index:
            for internal_name in entities_to_update['cranes']:
                pred = prediction_engine.predict_service_date(internal_name, 'crane', task_id)
                if pred: pred['entity_type'] = 'crane'
                all_preds.append(pred)

    # Predict for specific spreaders that were on cranes with new data
    if entities_to_update.get('spreaders'):
        spreader_tasks_df = config[config['category'] == 'Spreader']
        for task_id in spreader_tasks_df.index:
            for numeric_id in entities_to_update['spreaders']:
                pred = prediction_engine.predict_service_date(numeric_id, 'spreader', task_id)
                if pred: pred['entity_type'] = 'spreader'
                all_preds.append(pred)

    preds_df = pd.DataFrame([p for p in all_preds if p is not None and not p.get('error')])
    if not preds_df.empty:
        def map_id_to_display_name(row):
            if row['entity_type'] == 'crane': return INTERNAL_TO_DISPLAY_NAME.get(row['entity_id'], row['entity_id'])
            elif row['entity_type'] == 'spreader': return NUMERIC_ID_TO_DISPLAY_NAME.get(int(row['entity_id']), row['entity_id'])
            return row['entity_id']
        preds_df['entity_display_name'] = preds_df.apply(map_id_to_display_name, axis=1)
        logger.info(f"Successfully generated {len(preds_df)} targeted predictions.")
    return preds_df

# --- END OF PASTED BLOCK ---

@st.cache_data
def get_config(): return prediction_engine.load_service_config()

@st.cache_data(ttl=300)
def run_all_predictions():
    """Runs predictions for all cranes and all spreaders based on service_config."""
    logger.info("Executing run_all_predictions() for cranes and spreaders.")
    config = get_config()
    if config is None:
        return pd.DataFrame()

    all_preds = []
    
    # Get a list of all internal names for cranes from our new map
    crane_internal_names = [internal for display, internal in DISPLAY_TO_INTERNAL_NAME.items() if 'RMG' in internal]

    # --- Predict for Cranes ---
    crane_tasks_df = config[config['category'] != 'Spreader']
    for task_id in crane_tasks_df.index:
        for internal_name in crane_internal_names:
            pred = prediction_engine.predict_service_date(internal_name, 'crane', task_id)
            # FIX: This new part adds the 'crane' label
            if pred:
                pred['entity_type'] = 'crane'
            all_preds.append(pred)

        # --- Predict for Spreaders ---
    spreader_tasks_df = config[config['category'] == 'Spreader']
    # We use the display_to_numeric_id map to iterate through spreaders
    for display_name, numeric_id in DISPLAY_TO_NUMERIC_ID.items():
        for task_id in spreader_tasks_df.index:
            pred = prediction_engine.predict_service_date(numeric_id, 'spreader', task_id)
            # FIX: This new part adds the 'spreader' label
            if pred:
                pred['entity_type'] = 'spreader'
            all_preds.append(pred)

    preds_df = pd.DataFrame([p for p in all_preds if p is not None and not p.get('error')])

    if preds_df.empty:
        logger.warning("No predictions were generated.")
        return pd.DataFrame()

    # --- Add the user-friendly 'entity_display_name' column ---
    def map_id_to_display_name(row):
        if row['entity_type'] == 'crane':
            # Map internal name like 'RMG01' to 'CASC01 EAST'
            return INTERNAL_TO_DISPLAY_NAME.get(row['entity_id'], row['entity_id'])
        elif row['entity_type'] == 'spreader':
            # Map numeric id like 29336 to 'SP001'
            return NUMERIC_ID_TO_DISPLAY_NAME.get(int(row['entity_id']), row['entity_id'])
        return row['entity_id']

    preds_df['entity_display_name'] = preds_df.apply(map_id_to_display_name, axis=1)
    
    # For backwards compatibility and clarity, let's rename the old 'crane' column
    # if it exists, to our new standard 'entity_display_name'
    if 'crane' in preds_df.columns:
        preds_df.drop(columns=['crane'], inplace=True)

    logger.info(f"Successfully generated {len(preds_df)} predictions with display names.")
    return preds_df


if 'admin_logged_in' not in st.session_state: st.session_state['admin_logged_in'] = False
if 'admin_user' not in st.session_state: st.session_state['admin_user'] = None
if 'data_loaded' not in st.session_state: st.session_state['data_loaded'] = False
if 'active_tab' not in st.session_state: st.session_state['active_tab'] = "📊 Detailed Analysis"
if 'selected_spreader' not in st.session_state: st.session_state.selected_spreader = None
# Initialize session state for CSV import results
if 'valid_records' not in st.session_state: st.session_state['valid_records'] = []
if 'duplicate_records' not in st.session_state: st.session_state['duplicate_records'] = []
if 'error_records' not in st.session_state: st.session_state['error_records'] = []
# ---> PASTE THE NEW CODE HERE <---

# NEW: Add state for tracking updates
if 'all_preds_df' not in st.session_state:
    st.session_state.all_preds_df = pd.DataFrame()

if 'last_update_check' not in st.session_state:
    # Initialize with a time in the past to ensure the first check runs
    st.session_state.last_update_check = datetime.now(timezone.utc) - timedelta(days=1)


# FIX: Move database initialization to be the first step in the main app logic
# to prevent module loading race conditions.
# --- PASTE THIS NEW CODE IN ITS PLACE ---

# --- Main Application Flow ---

# Initialize the database and ensure tables are created.
database.init_db()

# Centralized data loading function with caching.
@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_data():
    """
    Loads all necessary data for the dashboard. It first tries to load predictions
    from the SQLite cache. If the cache is empty or stale, it runs the full
    prediction process and updates the cache.
    """
    with st.spinner("Performing initial data load and predictions... This may take a moment."):
        logger.info("Attempting to load all application data.")

        # Attempt to load predictions from the database cache first.
        predictions_df = database.get_all_predictions()

        # If the cache is empty, run the initial predictions and store them.
        if predictions_df.empty:
            logger.warning("Cache was empty. Running initial predictions.")
            predictions_df = run_initial_predictions()
            if not predictions_df.empty:
                database.store_predictions(predictions_df)
        
        # Load other data sources.
        service_config = get_config()
        all_logs_df = database.get_all_service_logs()
        all_windows_df = database.get_all_maintenance_windows()
        
        logger.info("All data successfully loaded.")
        return service_config, predictions_df, all_logs_df, all_windows_df

# --- Load data and assign to variables ---
service_config, all_preds_df, all_logs_df, all_windows_df = load_data()

# Check if data loading failed at a critical step.
if all_preds_df.empty or service_config is None:
    st.error("A critical error occurred during data loading. Please check the logs.")
    st.stop()

# --- END OF PASTED BLOCK ---

# --- Main App UI ---
st.title("🏗️ Crane & Spreader Maintenance Dashboard")

if service_config is None:
    st.error("`service_config.csv` not found.")
    st.stop()

# Get sorted lists of DISPLAY names from our new mapping function
CRANE_LIST = sorted([name for name in DISPLAY_TO_INTERNAL_NAME.keys() if 'RMG' in DISPLAY_TO_INTERNAL_NAME[name]])
SPREADER_LIST = sorted([name for name in DISPLAY_TO_INTERNAL_NAME.keys() if 'SP' in DISPLAY_TO_INTERNAL_NAME[name]])
ENTITY_LIST = CRANE_LIST + SPREADER_LIST

# We still need the old spreader maps for some specific functions, so let's keep this line
SPREADER_NAME_TO_ID, SPREADER_ID_TO_NAME = get_spreader_name_maps()

# --- Helper Function ---
def format_and_style_df(df):
    df_display = df.copy()
    if 'predicted_date' in df_display.columns and not df_display['predicted_date'].isnull().all():
        df_display['predicted_date'] = pd.to_datetime(df_display['predicted_date']).dt.strftime(Settings.DATE_FORMAT)
    if 'service_date' in df_display.columns and not df_display['service_date'].isnull().all():
        df_display['service_date'] = pd.to_datetime(df_display['service_date']).dt.strftime(Settings.DATE_FORMAT)
    if 'from_date' in df_display.columns and not df_display['from_date'].isnull().all():
        df_display['from_date'] = pd.to_datetime(df_display['from_date']).dt.strftime(Settings.DATE_FORMAT)
    if 'to_date' in df_display.columns and not df_display['to_date'].isnull().all():
        df_display['to_date'] = pd.to_datetime(df_display['to_date']).dt.strftime(Settings.DATE_FORMAT)
    return df_display

# --- Main Application Tabs ---
tab_options = [
    "❤️ Fleet Health",
    "📊 Detailed Analysis", 
    "Spreader Movement",
    "🗓️ Maintenance Overview", 
    "⚠️ Overdue Services", 
    "🧠 Smart Planner", 
    "📜 Maintenance Records",
    "⚙️ Admin"
]

try:
    active_tab_index = tab_options.index(st.session_state.active_tab)
except ValueError:
    logger.warning(f"Active tab '{st.session_state.active_tab}' not in options. Defaulting to first tab.")
    active_tab_index = 0
    st.session_state.active_tab = tab_options[0]

def on_tab_change():
    logger.info(f"Tab changed to: {st.session_state.tab_selector}")
    st.session_state.active_tab = st.session_state.tab_selector

selected_tab = st.radio(
    "Main Navigation", 
    options=tab_options, 
    key="tab_selector",
    horizontal=True,
    on_change=on_tab_change,
    label_visibility="collapsed",
    index=active_tab_index
)

# --- Tab Content ---
if selected_tab == "❤️ Fleet Health":
    st.header("❤️ Fleet Health at a Glance")

    def get_cumulative_health_style(tasks_df):
        """
        Calculates cumulative health and style for a group of tasks (Service or Inspection).
        Returns a CSS style string, a health score (0-1), and a status color.
        """
        if tasks_df.empty or tasks_df['service_interval_days'].sum() == 0:
            return "background: #333333;", 0, "#888888" # Default grey for no data

        # Calculate individual health scores (0 to 1 scale)
        # We clip to ensure health doesn't go below 0 or above 1.
        tasks_df['health_score'] = np.clip(
            tasks_df['days_remaining'] / tasks_df['service_interval_days'], 0, 1
        )
        
        # Calculate cumulative health as the average score
        cumulative_health = tasks_df['health_score'].mean()
        
        # Determine color based on the cumulative health score
        if cumulative_health <= 0.25:
            color = "#FF4500"  # Vibrant Red/Orange for poor health
        elif cumulative_health <= 0.6:
            color = "#FFD700"  # Vibrant Yellow for warning
        else:
            color = "#00FF7F"  # Vibrant Green (Spring Green) for healthy
        
        percentage = cumulative_health * 100
        style = f"background: conic-gradient({color} {percentage}%, #333333 0);"
        
        return style, cumulative_health, color

    def display_health_widgets(entity_type, predictions_df, config_df):
        st.subheader(f"{entity_type.capitalize()}s")
        entity_preds = predictions_df[predictions_df['entity_type'] == entity_type].copy()
        
        if entity_preds.empty:
            st.warning(f"No prediction data available for {entity_type}s.")
            return

        # --- FIX: Ensure 'action_required' column exists ---
        # If the data is loaded from a stale cache, this column might be missing.
        # This check merges it from the main config if necessary.
        if 'action_required' not in entity_preds.columns:
            logger.warning("'action_required' column not found in predictions, merging from config.")
            config_with_id = config_df.reset_index()[['task_id', 'action_required']]
            entity_preds = entity_preds.merge(config_with_id, on='task_id', how='left')

        # Clean up data for calculation
        entity_preds['service_interval_days'] = pd.to_numeric(entity_preds['service_interval_days'], errors='coerce').fillna(1)
        entity_preds.dropna(subset=['days_remaining', 'action_required'], inplace=True)
        
        # Define keywords for categorization
        inspection_keywords = ['check', 'inspect', 'measure', 'test', 'recalibrate']
        
        def categorize_task(action):
            action_lower = str(action).lower()
            return 'Inspection' if any(kw in action_lower for kw in inspection_keywords) else 'Service'

        entity_preds['task_category'] = entity_preds['action_required'].apply(categorize_task)

        entities = sorted(entity_preds['entity_display_name'].unique())
        num_columns = 4 # Adjust number of columns for layout if needed
        cols = st.columns(num_columns)

        for i, entity_name in enumerate(entities):
            with cols[i % num_columns]:
                entity_df = entity_preds[entity_preds['entity_display_name'] == entity_name]
                
                service_tasks = entity_df[entity_df['task_category'] == 'Service']
                inspection_tasks = entity_df[entity_df['task_category'] == 'Inspection']
                
                service_style, service_health, _ = get_cumulative_health_style(service_tasks)
                inspection_style, inspection_health, _ = get_cumulative_health_style(inspection_tasks)
                
                if service_health > 0 and inspection_health > 0:
                    overall_health = (service_health + inspection_health) / 2 * 10
                elif service_health > 0:
                    overall_health = service_health * 10
                else:
                    overall_health = inspection_health * 10
                
                tooltip = (f"Overall Health: {overall_health:.1f}/10 | "
                           f"Service: {service_health*100:.0f}% | "
                           f"Inspection: {inspection_health*100:.0f}%")

                min_health = min(service_health, inspection_health) if (service_health > 0 and inspection_health > 0) else (service_health or inspection_health)
                if min_health <= 0.25:
                    status_text, status_color = "Action Required", "#FF4500"
                elif min_health <= 0.6:
                    status_text, status_color = "Warning", "#FFD700"
                else:
                    status_text, status_color = "Healthy", "#00FF7F"

                st.markdown(f"""
                <div class="health-widget-container" title="{tooltip}">
                    <h4 style="color: #FFFFFF;">{entity_name}</h4>
                    <div class="rings-container">
                        <div class="health-ring outer-ring" style="{service_style}">
                            <div class="health-ring inner-ring" style="{inspection_style}"></div>
                        </div>
                    </div>
                    <p class="health-status" style="color:{status_color};">{status_text}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # --- Display Widgets ---
    # Pass the service_config DataFrame to the function
    display_health_widgets('crane', all_preds_df, service_config)
    st.divider()
    display_health_widgets('spreader', all_preds_df, service_config)


elif selected_tab == "📊 Detailed Analysis":
    st.sidebar.header("Analysis Selections")
    selected_entity = st.sidebar.selectbox("Select Crane or Spreader:", ENTITY_LIST, key="entity_select_tab1")

    # Initialize the variable to a safe default value right away
    selected_task_id = None # <--- ADD THIS LINE HERE

    if selected_entity and selected_entity.startswith("RMG"):
        conflicted_cranes = find_spreader_conflicts()
        if selected_entity in conflicted_cranes:
            st.warning("⚠️ **Data Conflict:** This crane is reporting the same Spreader ID as another crane. Spreader-related predictions may be unreliable.")

    # Check if service_config exists and is not empty before proceeding.
    # Note: Using st.session_state is not needed here as service_config is loaded globally.
    if service_config is not None and not service_config.empty:
        tasks_df = service_config

        # Filter tasks dynamically based on the 'category' column
        if selected_entity.startswith('SP'):
            tasks_df = tasks_df[tasks_df['category'] == 'Spreader']
        else: # It's a crane
            tasks_df = tasks_df[tasks_df['category'] != 'Spreader']

        # --- DEPENDENT DROPDOWN LOGIC ---
        if not tasks_df.empty:
            categories = sorted(tasks_df['category'].unique())
            selected_category = st.sidebar.selectbox(
                "Select Category:",
                options=categories,
                key="category_select"
            )

            if selected_category:
                category_tasks_df = tasks_df[tasks_df['category'] == selected_category]
                if not category_tasks_df.empty:
                    components = sorted(category_tasks_df['component'].unique())
                    selected_component = st.sidebar.selectbox(
                        "Select Component:",
                        options=components,
                        key="component_select"
                    )

                    if selected_component:
                        component_tasks_df = category_tasks_df[category_tasks_df['component'] == selected_component]
                        if not component_tasks_df.empty:
                            component_tasks_df = component_tasks_df.sort_values('action_required')
                            # This line now only runs if there are tasks to select
                            selected_task_id = st.sidebar.selectbox(
                                "Select Service or Inspection:",
                                options=component_tasks_df.index,
                                format_func=lambda task_id: component_tasks_df.loc[task_id, 'action_required'],
                                key="metric_select"
                            )

    # This check is now completely safe.
    if selected_task_id:
        selected_metric_info = service_config.loc[selected_task_id]
        st.subheader(f"Analysis for {selected_entity} - {selected_metric_info['action_required']}")
        
        is_usage_based = selected_metric_info['tag_name'] != '' and selected_metric_info['service_limit'] != ''

        if is_usage_based:
            st.sidebar.subheader("What-If Analysis")
            default_limit = float(selected_metric_info['service_limit']) if selected_metric_info['service_limit'] else 0.0
            custom_limit = st.sidebar.number_input(
                "Adjust Service Limit:", min_value=0.0, value=default_limit, step=50000.0,
                key=f"custom_limit_{selected_entity}_{selected_task_id}"
            )
        else:
            custom_limit = None

        # ... inside "Detailed Analysis" tab
        # Get the internal ID using our new map
        internal_id = DISPLAY_TO_INTERNAL_NAME.get(selected_entity)

        if internal_id and 'SP' in internal_id:
            entity_type = 'spreader'
            # For spreaders, the prediction engine needs the numeric ID
            entity_id = DISPLAY_TO_NUMERIC_ID.get(selected_entity)
        else:
            entity_type = 'crane'
            # For cranes, the prediction engine uses the internal name (e.g., 'RMG01')
            entity_id = internal_id
        
        if entity_id:
            prediction_result = prediction_engine.predict_service_date(entity_id, entity_type, selected_task_id, custom_limit)
        else:
            prediction_result = {'error': f"Could not find ID for {selected_entity}"}

        st.info(f"**Action Required:** {prediction_result.get('action_required', 'N/A')}")
        st.write("---") 
        col1, col2, col3 = st.columns(3)
        time_limit_days = prediction_result.get('service_interval_days')
        usage_limit = prediction_result.get('service_limit')
        
        with col1:
            if is_usage_based:
                st.metric("Value Since Last Service", f"{prediction_result.get('current_value', 0):,.0f} {prediction_result.get('unit', '')}")
            else:
                last_service_date_obj = prediction_result.get('last_service_date')
                last_service_str = last_service_date_obj.strftime(Settings.DATE_FORMAT) if pd.notna(last_service_date_obj) else "N/A"
                st.metric("Last Serviced On", last_service_str)

        with col2:
            limit_str_parts = []
            if pd.notna(usage_limit) and usage_limit > 0:
                limit_str_parts.append(f"{usage_limit:,.0f} {prediction_result.get('unit', '')}")
            if pd.notna(time_limit_days) and time_limit_days > 0:
                months = round(time_limit_days / 30)
                limit_str_parts.append(f"{months} mths")
            limit_display_str = " / ".join(limit_str_parts) if limit_str_parts else "N/A"
            st.metric("Service Limit", limit_display_str)

        with col3:
            if is_usage_based:
                st.metric("Avg. Daily Usage", f"{prediction_result.get('avg_daily_usage', 0):,.2f} {prediction_result.get('unit', '')}/day")
            elif pd.notna(time_limit_days):
                last_service_date_obj = prediction_result.get('last_service_date')
                if pd.notna(last_service_date_obj):
                    elapsed_days = (datetime.now() - last_service_date_obj).days
                    st.metric("Time Elapsed", f"{elapsed_days} days")
                else:
                    st.metric("Time Elapsed", "N/A")

        if is_usage_based and pd.notna(usage_limit) and usage_limit > 0:
            progress_val = prediction_result.get('current_value', 0)
            st.progress(min(1.0, progress_val / usage_limit))
        elif not is_usage_based and pd.notna(time_limit_days) and time_limit_days > 0:
            last_service_date_obj = prediction_result.get('last_service_date')
            if pd.notna(last_service_date_obj):
                elapsed_days = (datetime.now() - last_service_date_obj).days
                st.progress(min(1.0, elapsed_days / time_limit_days))

        st.divider()
        pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)
        predicted_date_obj = prediction_result.get('predicted_date')
        predicted_date_str = predicted_date_obj.strftime(Settings.DATE_FORMAT) if pd.notna(predicted_date_obj) else "N/A"
        pred_col1.metric("Predicted Service Date", predicted_date_str)
        days_remaining_val = prediction_result.get('days_remaining')
        days_remaining_str = f"{days_remaining_val} days" if days_remaining_val is not None else "N/A"
        pred_col2.metric("Days Remaining", days_remaining_str)
        due_reason_val = prediction_result.get('due_reason', 'N/A')
        pred_col3.metric("Due Reason", due_reason_val)
        duration_val = prediction_result.get('duration_hours', 'N/A')
        duration_str = f"{duration_val} hours" if pd.notna(duration_val) and duration_val != '' else "N/A"
        pred_col4.metric("Est. Duration", duration_str)
        
        if prediction_result.get('error'):
            st.warning(f"**Prediction Issue:** {prediction_result['error']}")

        st.divider()
        
        if is_usage_based:
            st.subheader("Full Historical Data")
            tag_name_for_history = selected_metric_info['tag_name']
            
            if entity_type == 'crane':
                full_history_df = prediction_engine.get_full_history_for_metric(entity_id, tag_name_for_history)
            elif entity_type == 'spreader':
                full_history_df = prediction_engine.get_spreader_usage_history(entity_id, tag_name_for_history)
            else:
                full_history_df = pd.DataFrame()

            if not full_history_df.empty:
                line = alt.Chart(full_history_df).mark_line(interpolate='step-after', point=alt.OverlayMarkDef(color="#1f77b4", size=25)).encode(x=alt.X('timestamp:T', title='Timestamp'), y=alt.Y('tag_value:Q', title='Usage Value', scale=alt.Scale(zero=False)), tooltip=[alt.Tooltip('timestamp:T', title='Time'), alt.Tooltip('tag_value:Q', title='Value', format=",.0f")]).properties(title=f"Historical Data for {tag_name_for_history}")
                nearest = alt.selection_point(name='nearest', on='mouseover', fields=['timestamp'], empty=True)
                selectors = alt.Chart(full_history_df).mark_point().encode(x='timestamp:T', opacity=alt.value(0)).add_params(nearest)
                points = line.mark_point().encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
                text = line.mark_text(align='left', dx=5, dy=-5, fontSize=14).encode(text=alt.condition(nearest, 'tag_value:Q', alt.value(' '), format=",.0f"))
                rules = alt.Chart(full_history_df).mark_rule(color='gray').encode(x='timestamp:T').transform_filter(nearest)
                chart_layers = [line, selectors, points, text, rules]
                
                service_log_df_metric = database.get_all_service_logs_for_task(entity_id, entity_type, selected_task_id)
                if not service_log_df_metric.empty:
                    service_log_df_metric['service_date'] = pd.to_datetime(service_log_df_metric['service_date'])
                    service_log_df_metric['legend'] = 'Logged Service'
                    service_lines = alt.Chart(service_log_df_metric).mark_rule(strokeDash=[5,5], size=2).encode(x='service_date:T', color=alt.Color('legend:N', scale=alt.Scale(domain=['Logged Service'], range=['red']), title="Events"), tooltip=[alt.Tooltip('service_date:T', title='Service Date', format=Settings.DATE_FORMAT), alt.Tooltip('serviced_at_value:Q', title='Value at Service', format=",.0f"), alt.Tooltip('serviced_by:N', title='Serviced By')])
                    chart_layers.append(service_lines)

                final_chart = alt.layer(*chart_layers).interactive().configure_axis(labelFontSize=12, titleFontSize=14).configure_title(fontSize=16, anchor='start').configure_legend(titleFontSize=12, labelFontSize=11, orient='top-right')
                st.altair_chart(final_chart, use_container_width=True, theme="streamlit")
            else:
                st.write("No historical data to display for this metric.")
        else:
            st.info("This is a time-based task. No historical usage data to display.")
    else:
        st.info("Please select a service or inspection from the sidebar.")

elif selected_tab == "Spreader Movement":
    st.header("Interactive Fleet Dashboard")
    spreader_statuses_df = get_all_spreader_statuses()
    if not spreader_statuses_df.empty:
        st.subheader("Fleet Status Grid")
        num_columns = 4
        cols = st.columns(num_columns)
        
        # MODIFICATION: Sort the DataFrame by spreader name numerically to ensure correct order.
        spreader_statuses_df['sort_key'] = spreader_statuses_df['spreader_name'].str.extract('(\d+)').astype(int)
        spreader_statuses_df = spreader_statuses_df.sort_values(by='sort_key').reset_index(drop=True)
        
        for index, row in spreader_statuses_df.iterrows():
            col = cols[index % num_columns]
            with col:
                card_class = "spreader-card conflicted-card" if row['is_conflicted'] else "spreader-card"
                last_seen_str = row['last_seen_date'].strftime(Settings.DATE_FORMAT) if isinstance(row['last_seen_date'], (datetime, pd.Timestamp)) else 'N/A'
                
                # Use the display name for the location
                display_location = INTERNAL_TO_DISPLAY_NAME.get(row['crane_number'], row['crane_number'])
                
                card_html = f"""
                    <div class="{card_class}">
                        <b>{row['spreader_name']}</b><br>
                        Location: {display_location}
                        <div class="hover-details">
                            Last Seen: {last_seen_str}<br>
                            Status: {'Conflict' if row['is_conflicted'] else row['status']}
                        </div>
                    </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                if st.button("View Timeline", key=f"view_{row['spreader_name']}"):
                    st.session_state.selected_spreader = row['spreader_name']
    else:
        st.info("No spreader status data available.")

    # --- THIS ENTIRE BLOCK IS REPLACED WITH THE CORRECTED LOGIC ---
    st.divider()
    if st.session_state.selected_spreader:
        spreader_id = SPREADER_NAME_TO_ID.get(st.session_state.selected_spreader)
        if spreader_id:
            history_df = database.get_spreader_location_history(spreader_id)
            
            if not history_df.empty:
                # 1. CREATE A NEW 'Crane' COLUMN WITH THE CORRECT DISPLAY NAMES
                # This line translates 'RMG01' to 'CASC01 EAST' using the global map
                history_df['Crane'] = history_df['crane_number'].map(INTERNAL_TO_DISPLAY_NAME).fillna(history_df['crane_number'])

                # 2. DISPLAY THE DATA TABLE WITH CORRECT NAMES
                st.subheader(f"Spreader Movement History for {st.session_state.selected_spreader}")
                st.dataframe(
                    history_df[['Crane', 'from_date', 'to_date', 'duration_days']],
                    use_container_width=True,
                    column_config={
                        "Crane": st.column_config.TextColumn("Crane"),
                        "from_date": st.column_config.DatetimeColumn("From", format="DD/MM/YYYY HH:mm"),
                        "to_date": st.column_config.DatetimeColumn("To", format="DD/MM/YYYY HH:mm"),
                        "duration_days": st.column_config.NumberColumn("Duration (Days)", format="%.2f")
                    },
                    hide_index=True,
                )

                # 3. DISPLAY THE TIMELINE CHART USING THE NEW 'Crane' COLUMN
                st.subheader(f"Timeline for {st.session_state.selected_spreader}")
                # Use the new 'Crane' column for the domain, labels, and colors
                crane_domain = sorted(history_df['Crane'].unique()) 
                color_range = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78']
                
                chart = alt.Chart(history_df).mark_bar(cornerRadius=5, height=25).encode(
                    x=alt.X('from_date:T', title='Timeline'), 
                    x2='to_date:T', 
                    y=alt.Y('Crane:N', title='Crane', sort='-x'), 
                    color=alt.Color('Crane:N', scale=alt.Scale(domain=crane_domain, range=color_range[:len(crane_domain)]), legend=None),
                    tooltip=[
                        alt.Tooltip('Crane', title='Crane'), 
                        alt.Tooltip('from_date:T', title='From', format='%Y-%m-%d %H:%M'), 
                        alt.Tooltip('to_date:T', title='To', format='%Y-%m-%d %H:%M'), 
                        alt.Tooltip('duration_days:Q', title='Duration (days)', format='.1f')
                    ]
                ).properties(height=alt.Step(40)).interactive()
                
                st.altair_chart(chart, use_container_width=True, theme="streamlit")
            else:
                st.warning(f"No history found for {st.session_state.selected_spreader}.")
        else:
            st.error(f"Could not find ID for {st.session_state.selected_spreader}.")
    else:
        st.info("Click 'View Timeline' on a card above to see its detailed history.")
    # --- END OF REPLACED BLOCK ---

elif selected_tab == "🗓️ Maintenance Overview":
    st.header("Upcoming Maintenance Schedule")
    if not all_preds_df.empty:
        predictable_df = all_preds_df.dropna(subset=['days_remaining']).copy()
        predictable_df['days_remaining'] = predictable_df['days_remaining'].astype(int)
        overview_days = st.slider("Show services due within the next (days):", 7, 180, 60, 7, key="overview_slider")
        upcoming_df = predictable_df[(predictable_df['days_remaining'] <= overview_days) & (predictable_df['days_remaining'] > 0)]
        if not upcoming_df.empty:
                        # ... inside "Maintenance Overview" tab
            display_cols = ['entity_display_name', 'action_required', 'predicted_date', 'days_remaining', 'due_reason']
            display_df = format_and_style_df(upcoming_df[display_cols].sort_values('days_remaining'))
            st.dataframe(display_df.rename(columns={'entity_display_name': 'Entity', 'action_required': 'Task'}), use_container_width=True, height=600)
        else:
            st.success(f"No services are due within the next {overview_days} days.")
    else:
        st.warning("No prediction data available.")

elif selected_tab == "⚠️ Overdue Services":
    st.header("Overdue Services")
    if not all_preds_df.empty:
        predictable_df = all_preds_df.dropna(subset=['days_remaining']).copy()
        predictable_df['days_remaining'] = predictable_df['days_remaining'].astype(int)
        overdue_df = predictable_df[predictable_df['days_remaining'] <= 0]
        if not overdue_df.empty:
                        # ... inside "Overdue Services" tab
            display_cols = ['entity_display_name', 'action_required', 'predicted_date', 'days_remaining', 'due_reason']
            display_df = format_and_style_df(overdue_df[display_cols].sort_values('days_remaining'))
            st.dataframe(display_df.rename(columns={'entity_display_name': 'Entity', 'action_required': 'Task'}), use_container_width=True, height=600)
        else:
            st.success("No services are currently overdue.")
    else:
        st.warning("No prediction data available.")

elif selected_tab == "🧠 Smart Planner":
    st.header("Smart Maintenance Planner")
    if not all_preds_df.empty:
        predictable_df = all_preds_df.dropna(subset=['days_remaining']).copy()
        predictable_df['days_remaining'] = predictable_df['days_remaining'].astype(int)
        planner_window = st.slider("Group services due within a window of (days):", 3, 45, 14, 1, key="planner_slider")
        future_services = predictable_df[predictable_df['days_remaining'] > 0].sort_values(by='predicted_date').copy()
        
        if not future_services.empty:
            future_services['predicted_date_dt'] = pd.to_datetime(future_services['predicted_date'], errors='coerce')
            for entity_name, group_df in future_services.groupby('entity_display_name'):
                if len(group_df) > 1:
                    group_df['date_diff'] = group_df['predicted_date_dt'].diff().dt.days.fillna(0)
                    clusters = (group_df['date_diff'] > planner_window).cumsum()
                    for _, cluster_df in group_df.groupby(clusters):
                        if len(cluster_df) > 1:
                            with st.expander(f"**Recommendation for {entity_name}** - Service cluster found!"):
                                first_due_date = cluster_df['predicted_date_dt'].min().strftime(Settings.DATE_FORMAT)
                                st.write(f"Consider servicing these items together around **{first_due_date}**.")
                                display_df = format_and_style_df(cluster_df)
                                st.dataframe(display_df[['action_required', 'predicted_date', 'days_remaining']], use_container_width=True)
        else:
            st.info("No upcoming services available for planning.")
    else:
        st.warning("No prediction data available for planning.")

elif selected_tab == "📜 Maintenance Records":
    st.header("Maintenance Records")
    view_selection = st.radio("Select View:", ["Service Log", "Maintenance Windows"], horizontal=True, label_visibility="collapsed")

    if view_selection == "Service Log":
        st.subheader("Complete Service History Log")
        if not all_logs_df.empty:
            st.write("Use the filters below to search the service history.")
            filter_col1, filter_col2 = st.columns(2)
            task_id_to_action = service_config['action_required'].to_dict()
            logs_with_names_df = all_logs_df.copy()
            # ... inside "Maintenance Records" tab / "Service Log" view
            def get_entity_name(row):
                if row['entity_type'] == 'spreader':
                    try:
                        # Map numeric ID to display name
                        return SPREADER_ID_TO_NAME.get(int(row['entity_id']), row['entity_id'])
                    except (ValueError, TypeError):
                        return row['entity_id']
                else:
                    # Map internal name to display name
                    return INTERNAL_TO_DISPLAY_NAME.get(row['entity_id'], row['entity_id'])

            logs_with_names_df['entity_name'] = logs_with_names_df.apply(get_entity_name, axis=1)

            with filter_col1:
                entity_options = ["All"] + sorted(logs_with_names_df['entity_name'].unique().tolist())
                selected_entities = st.multiselect("Filter by Crane/Spreader:", entity_options, default=["All"])
            
            with filter_col2:
                task_options = ["All"] + sorted(logs_with_names_df['task_id'].unique().tolist())
                selected_tasks = st.multiselect("Filter by Task:", options=task_options, format_func=lambda task_id: task_id_to_action.get(task_id, task_id), default=["All"])
            
            filtered_df = logs_with_names_df.copy()
            if "All" not in selected_entities: filtered_df = filtered_df[filtered_df['entity_name'].isin(selected_entities)]
            if "All" not in selected_tasks: filtered_df = filtered_df[filtered_df['task_id'].isin(selected_tasks)]
            
            filtered_df['Task'] = filtered_df['task_id'].map(task_id_to_action)
            display_cols = ['entity_name', 'Task', 'service_date', 'serviced_at_value', 'serviced_by']
            st.dataframe(format_and_style_df(filtered_df[display_cols].rename(columns={'entity_name': 'Entity'})), use_container_width=True, height=800)
        else:
            st.info("No service history has been logged in the database yet.")
    
    else: # Maintenance Windows View
        st.subheader("Scheduled Maintenance Windows")
        if not all_windows_df.empty and 'entity_id' in all_windows_df.columns:
            display_windows_df = all_windows_df.copy()
            
            def get_window_entity_name(row):
                if row['entity_type'] == 'spreader':
                    try:
                        return SPREADER_ID_TO_NAME.get(int(row['entity_id']), row['entity_id'])
                    except (ValueError, TypeError):
                        return row['entity_id']
                else:
                    return INTERNAL_TO_DISPLAY_NAME.get(row['entity_id'], row['entity_id'])
            
            display_windows_df['entity_name'] = display_windows_df.apply(get_window_entity_name, axis=1)

            # Create formatted string columns for display
            display_windows_df['from_str'] = display_windows_df['from_datetime'].dt.strftime('%d/%m/%y %H:%M')
            display_windows_df['to_str'] = display_windows_df['to_datetime'].dt.strftime('%d/%m/%y %H:%M')

            # Define the columns to display and their friendly names
            display_cols = [
                'entity_name', 
                'from_str', 
                'to_str', 
                'service_type', 
                'task_description', 
                'notes'
            ]
            column_config = {
                "entity_name": "Entity",
                "from_str": "From",
                "to_str": "To",
                "service_type": "Service Type",
                "task_description": "Task Description",
                "notes": "Notes"
            }

            st.dataframe(
                display_windows_df[display_cols], 
                use_container_width=True, 
                column_config=column_config, 
                height=800
            )
        else:
            st.info("No maintenance windows are currently scheduled.")

elif selected_tab == "⚙️ Admin":
    st.header("⚙️ Admin Panel")

    def clear_import_results():
        """Clears previous CSV import results from the session state."""
        keys_to_clear = ['valid_records', 'duplicate_records', 'error_records', 'last_uploaded_filename']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
    
    if not st.session_state.admin_logged_in:
        st.write("Enter admin credentials to manage the application.")
        with st.form("admin_login_form"):
            username = st.text_input("Admin Username")
            password = st.text_input("Admin Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                admin_user = auth.verify_user(username, password)
                if admin_user:
                    st.session_state.admin_logged_in = True
                    st.session_state.admin_user = admin_user
                    st.session_state.force_data_reload = True
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")
    else:
        st.success(f"Logged in as: **{st.session_state.admin_user}**")
        def logout_callback():
            st.session_state.admin_logged_in = False
            st.session_state.admin_user = None
            st.session_state.data_loaded = False
            st.session_state.active_tab = "📊 Detailed Analysis"
            if 'admin_action_message' in st.session_state: del st.session_state.admin_action_message
        
        def add_window_callback():
            selected_entities = st.session_state.get("window_entity_select", [])
            from_dt = datetime.combine(st.session_state.from_date, st.session_state.from_time)
            to_dt = datetime.combine(st.session_state.to_date, st.session_state.to_time)

            service_type = st.session_state.get("window_service_type")
            task_description = st.session_state.get("window_task_desc")
            notes = st.session_state.get("window_notes")

            if not selected_entities:
                st.session_state.admin_action_message = ("warning", "Please select at least one crane or spreader.")
                return
            if to_dt <= from_dt:
                st.session_state.admin_action_message = ("warning", "'To' date/time must be after 'From' date/time.")
                return

            success_adds = []
            conflict_adds = []
            
            for entity_display_name in selected_entities:
                internal_id = DISPLAY_TO_INTERNAL_NAME.get(entity_display_name)
                
                if not internal_id:
                    logger.error(f"Could not find internal ID for '{entity_display_name}' in add_window_callback.")
                    continue

                entity_type = 'spreader' if 'SP' in internal_id else 'crane'
                entity_id_for_db = DISPLAY_TO_NUMERIC_ID.get(entity_display_name) if entity_type == 'spreader' else internal_id

                if database.check_for_conflicting_maintenance_windows(entity_id_for_db, from_dt, to_dt):
                    conflict_adds.append(entity_display_name)
                    continue

                if database.add_maintenance_window(entity_id_for_db, entity_type, from_dt, to_dt, service_type, task_description, notes):
                    success_adds.append(entity_display_name)

            msg_parts = []
            if success_adds:
                msg_parts.append(f"Successfully added windows for: {', '.join(success_adds)}.")
                st.session_state.all_windows_df = database.get_all_maintenance_windows()
                st.session_state.force_data_reload = True
            if conflict_adds:
                msg_parts.append(f"Skipped due to conflicts: {', '.join(conflict_adds)}.")
            
            if not msg_parts:
                st.session_state.admin_action_message = ("error", "Failed to add window for any selected items.")
            else:
                final_message = " ".join(msg_parts)
                msg_type = "success" if success_adds and not conflict_adds else "warning"
                st.session_state.admin_action_message = (msg_type, final_message)
        
        
        def delete_selected_logs_callback():
            log_keys_to_delete = st.session_state.get("delete_log_multiselect", [])
            if not log_keys_to_delete:
                st.session_state.admin_action_message = ("warning", "Please select at least one record to delete.")
                return
            
            delete_count = 0
            ids_to_delete = []
            for key in log_keys_to_delete:
                log_id = int(key.split(':')[0])
                if database.delete_service_log(log_id):
                    delete_count += 1
                    ids_to_delete.append(log_id)

            st.session_state.admin_action_message = ("success", f"Successfully deleted {delete_count} record(s).")
            st.session_state.all_logs_df = st.session_state.all_logs_df[~st.session_state.all_logs_df['id'].isin(ids_to_delete)]
            st.session_state.force_data_reload = True

        def delete_selected_windows_callback():
            window_keys_to_delete = st.session_state.get("delete_window_multiselect", [])
            if not window_keys_to_delete:
                st.session_state.admin_action_message = ("warning", "Please select at least one window to delete.")
                return
            
            delete_count = 0
            ids_to_delete = []
            for key in window_keys_to_delete:
                window_id = int(key.split(':')[0])
                if database.delete_maintenance_window(window_id):
                    delete_count += 1
                    ids_to_delete.append(window_id)
            st.session_state.admin_action_message = ("success", f"Successfully deleted {delete_count} window(s).")
            st.session_state.all_windows_df = st.session_state.all_windows_df[~st.session_state.all_windows_df['id'].isin(ids_to_delete)]


        def import_csv_records_callback():
            """
            Takes the validated records from the session state and imports them
            into the maintenance_windows table in the database.
            """
            valid_records_to_import = st.session_state.get('valid_records', [])
            if not valid_records_to_import:
                st.session_state.admin_action_message = ("warning", "No valid records to import.")
                return

            import_count = 0
            for record in valid_records_to_import:
                success = database.add_maintenance_window(
                    entity_id=record['entity_id'],
                    entity_type=record['entity_type'],
                    from_dt=record['from_datetime'],
                    to_dt=record['to_datetime'],
                    service_type=record.get('service_type'),
                    task_description=record.get('task_description'),
                    notes=record.get('notes')
                )
                if success:
                    import_count += 1

            if import_count > 0:
                st.session_state.admin_action_message = ("success", f"Successfully imported {import_count} maintenance windows.")
                st.session_state.force_data_reload = True
            else:
                st.session_state.admin_action_message = ("error", "Failed to import any records.")

            clear_import_results()


        if 'admin_action_message' in st.session_state:
            msg_type, msg_text = st.session_state.admin_action_message
            if msg_type == "success": st.success(msg_text)
            elif msg_type == "warning": st.warning(msg_text)
            else: st.error(msg_text)
            del st.session_state.admin_action_message

        admin_mode = st.radio("Select Management Area:", ["Service Records", "Maintenance Windows", "User Management"], horizontal=True, key='admin_mode_selector')
        st.divider()

        if admin_mode == "Service Records":
            st.subheader("Manage Service Records")
            st.write("**Log a New Completed Service**")
            admin_col1, admin_col2 = st.columns(2)
            with admin_col1:
                st.selectbox("Crane or Spreader:", ENTITY_LIST, key="admin_entity_select")
            with admin_col2:
                if 'service_config' in st.session_state and not st.session_state.service_config.empty:
                    selected_entity_admin = st.session_state.admin_entity_select
                    admin_tasks_df = st.session_state.service_config

                    is_spreader = any(sp_name in selected_entity_admin for sp_name in SPREADER_LIST)
                    if is_spreader:
                        admin_tasks_df = admin_tasks_df[admin_tasks_df['category'] == 'Spreader']
                    else:
                        admin_tasks_df = admin_tasks_df[admin_tasks_df['category'] != 'Spreader']
                    
                    if not admin_tasks_df.empty:
                        categories = sorted(admin_tasks_df['category'].unique())
                        selected_admin_category = st.selectbox("Service Category:", options=categories, key="admin_category_select")

                        if selected_admin_category:
                            category_df = admin_tasks_df[admin_tasks_df['category'] == selected_admin_category]
                            components = sorted(category_df['component'].unique())
                            selected_admin_component = st.selectbox("Service Component:", options=components, key="admin_component_select")

                            if selected_admin_component:
                                component_df = category_df[category_df['component'] == selected_admin_component].sort_values('action_required')
                                st.session_state.admin_task_select = st.selectbox(
                                    "Service Type:", 
                                    options=component_df.index, 
                                    format_func=lambda task_id: component_df.loc[task_id, 'action_required'], 
                                    key="admin_task_select_new"
                                )
                            else: st.session_state.admin_task_select = None
                        else: st.session_state.admin_task_select = None
                    else: st.session_state.admin_task_select = None

            selected_task_id = st.session_state.get("admin_task_select")
            selected_entity = st.session_state.admin_entity_select
            
            if selected_task_id:
                task_info = service_config.loc[selected_task_id]
                tag_name = task_info['tag_name']
                unit = task_info['unit']
                is_time_based = pd.isna(tag_name) or str(tag_name).strip() == ''
                value_input_label = "Value at Time of Service" + (f" ({unit})" if not is_time_based and pd.notna(unit) and str(unit).strip() != '' else "")
                default_duration = 0.0
                try:
                    duration_val = task_info['duration_hours']
                    if pd.notna(duration_val) and str(duration_val).strip() != '': default_duration = float(duration_val)
                except (ValueError, TypeError): logger.warning(f"Could not convert duration '{duration_val}' to float for task {selected_task_id}. Defaulting to 0.0.")
                
                default_service_value = 0.0
                if not is_time_based:
                    entity_id_for_hist = DISPLAY_TO_INTERNAL_NAME.get(selected_entity)
                    if 'SP' in entity_id_for_hist:
                        entity_id_for_hist = DISPLAY_TO_NUMERIC_ID.get(selected_entity)
                        history_df = prediction_engine.get_spreader_usage_history(entity_id_for_hist, tag_name) if entity_id_for_hist else pd.DataFrame()
                    else:
                        history_df = prediction_engine.get_full_history_for_metric(entity_id_for_hist, tag_name)
                    
                    if not history_df.empty:
                        last_value = history_df['tag_value'].iloc[-1]
                        if pd.notna(last_value): default_service_value = float(last_value)

                with admin_col1: st.date_input("Date of Service:", datetime.now(), key="admin_date_select")
                with admin_col2: st.number_input("Duration of Service (hours):", min_value=0.0, step=0.5, value=default_duration, key="admin_duration")
                st.number_input(label=value_input_label, min_value=0.0, value=default_service_value, format="%.2f", key="admin_value", disabled=is_time_based)

                if st.button("Log Service Record"):
                    internal_id = DISPLAY_TO_INTERNAL_NAME.get(selected_entity)
                    entity_type = 'spreader' if 'SP' in internal_id else 'crane'
                    entity_id_for_db = DISPLAY_TO_NUMERIC_ID.get(selected_entity) if entity_type == 'spreader' else internal_id
                    
                    if entity_id_for_db:
                        success = database.log_service_completed(entity_id_for_db, entity_type, st.session_state.admin_task_select, st.session_state.admin_date_select, st.session_state.admin_value if not is_time_based else 0.0, st.session_state.admin_user, st.session_state.admin_duration)
                        if success: st.session_state.admin_action_message = ("success", "Service logged successfully!"); st.session_state.force_data_reload = True; st.rerun()
                        else: st.session_state.admin_action_message = ("error", "Failed to log service."); st.rerun()
                    else:
                        st.session_state.admin_action_message = ("error", f"Could not find ID for {selected_entity}."); st.rerun()

            st.divider()
            st.write("**Existing Service Logs**")
            if not all_logs_df.empty:
                display_df = all_logs_df.copy()
                
                def map_log_name(row):
                    if row['entity_type'] == 'spreader':
                        return SPREADER_ID_TO_NAME.get(int(row['entity_id']), row['entity_id'])
                    return INTERNAL_TO_DISPLAY_NAME.get(row['entity_id'], row['entity_id'])
                
                display_df['entity_name'] = display_df.apply(map_log_name, axis=1)
                display_df['Task'] = display_df['task_id'].map(service_config['action_required'].to_dict())
                display_df['service_date_str'] = pd.to_datetime(display_df['service_date']).dt.strftime('%d/%m/%Y')
                
                st.dataframe(display_df[['entity_name', 'Task', 'service_date_str', 'serviced_at_value', 'duration_hours', 'serviced_by']], use_container_width=True, column_config={"entity_name": "Entity", "service_date_str": "Service Date", "serviced_at_value": "Value at Service", "duration_hours": "Duration (h)", "serviced_by": "Serviced By"})
                
                with st.expander("Delete Service Records"):
                    log_options_keys = [f"{row['id']}: {row['entity_name']} - {row['Task']} on {row['service_date_str']}" for _, row in display_df.iterrows()]
                    st.multiselect("Select service records to delete:", options=log_options_keys, key="delete_log_multiselect")
                    st.button("Delete Selected Service Records", type="primary", on_click=delete_selected_logs_callback)
            else: st.info("No service logs found.")

        elif admin_mode == "Maintenance Windows":
            st.subheader("Manage Maintenance Windows")
            
            with st.form("add_window_form", clear_on_submit=True):
                st.write("**Log a New Maintenance Window**")
                st.multiselect("Crane(s) or Spreader(s):", ENTITY_LIST, key="window_entity_select")
                
                col_a, col_b = st.columns(2)
                with col_a: st.text_input("Service Type:", key="window_service_type", placeholder="e.g., PM, CM, INSP")
                with col_b: st.text_input("Task Description (Service Alias):", key="window_task_desc", placeholder="e.g., PM-CR-01")
                st.text_area("Repairer Notes:", key="window_notes", placeholder="Add any relevant notes here...")

                from_col, to_col = st.columns(2)
                with from_col: 
                    st.date_input("From Date", datetime.now(), key="from_date")
                    st.time_input("From Time", datetime.now().time(), key="from_time", step=1800)
                with to_col: 
                    st.date_input("To Date", datetime.now(), key="to_date")
                    st.time_input("To Time", (datetime.now() + timedelta(hours=4)).time(), key="to_time", step=1800)
                    
                st.form_submit_button("Add Maintenance Window", on_click=add_window_callback)

            st.divider()
            st.subheader("Import Plan from CSV")
            
            uploaded_file = st.file_uploader(
                "Upload a maintenance plan CSV file.", type="csv", key="csv_uploader",
                help="The CSV must contain the columns: 'Fleet #', 'Vehicle Type', 'Scheduled Start Date', 'Scheduled End Date', 'Service Type', 'Service Alias', 'Repairer Notes'.",
                on_change=clear_import_results
            )

            if uploaded_file is not None:
                try:
                    if 'last_uploaded_filename' not in st.session_state or st.session_state.last_uploaded_filename != uploaded_file.name:
                        st.session_state.last_uploaded_filename = uploaded_file.name
                        df = pd.read_csv(uploaded_file)
                        valid, duplicates, errors = _validate_maintenance_plan(df)
                        st.session_state.valid_records = valid
                        st.session_state.duplicate_records = duplicates
                        st.session_state.error_records = errors
                    
                    if st.session_state.get('valid_records'):
                        with st.expander(f"✅ Ready to Import ({len(st.session_state.valid_records)} records)", expanded=True):
                            display_valid_df = pd.DataFrame(st.session_state.valid_records)
                            display_valid_df['from_datetime'] = pd.to_datetime(display_valid_df['from_datetime']).dt.strftime('%d/%m/%Y %H:%M')
                            display_valid_df['to_datetime'] = pd.to_datetime(display_valid_df['to_datetime']).dt.strftime('%d/%m/%Y %H:%M')
                            st.dataframe(display_valid_df, use_container_width=True)
                            st.button("Import Valid Records to Database", type="primary", on_click=import_csv_records_callback, use_container_width=True)

                    if st.session_state.get('duplicate_records'):
                        with st.expander(f"⚠️ Duplicates - Will be Skipped ({len(st.session_state.duplicate_records)} records)"):
                            st.dataframe(pd.DataFrame(st.session_state.duplicate_records), use_container_width=True)

                    if st.session_state.get('error_records'):
                        with st.expander(f"❌ Rows with Errors ({len(st.session_state.error_records)} records)"):
                            st.dataframe(pd.DataFrame(st.session_state.error_records), use_container_width=True)

                except Exception as e:
                    st.error(f"An error occurred while processing the file: {e}")
                    logger.error(f"CSV processing failed: {e}", exc_info=True)

            st.divider()
            st.write("**Existing Maintenance Windows**")
            if not all_windows_df.empty and 'entity_id' in all_windows_df.columns:
                display_windows_df = all_windows_df.copy()
                
                def get_window_entity_name_admin(row):
                    if row['entity_type'] == 'spreader':
                        try: return SPREADER_ID_TO_NAME.get(int(row['entity_id']), row['entity_id'])
                        except (ValueError, TypeError): return row['entity_id']
                    else: return INTERNAL_TO_DISPLAY_NAME.get(row['entity_id'], row['entity_id'])
                
                display_windows_df['entity_name'] = display_windows_df.apply(get_window_entity_name_admin, axis=1)
                display_windows_df['from_str'] = display_windows_df['from_datetime'].dt.strftime('%d/%m/%y %H:%M')
                display_windows_df['to_str'] = display_windows_df['to_datetime'].dt.strftime('%d/%m/%y %H:%M')

                display_cols = ['entity_name', 'from_str', 'to_str', 'service_type', 'task_description', 'notes']
                column_config = {"entity_name": "Entity", "from_str": "From", "to_str": "To", "service_type": "Service Type", "task_description": "Task Description", "notes": "Notes"}

                st.dataframe(display_windows_df[display_cols], use_container_width=True, column_config=column_config, height=800)
                
                with st.expander("Delete Maintenance Windows"):
                    window_options_keys = [f"{row['id']}: {row['entity_name']} from {row['from_str']}" for _, row in display_windows_df.iterrows()]
                    st.multiselect("Select maintenance windows to delete:", options=window_options_keys, key="delete_window_multiselect")
                    st.button("Delete Selected Windows", type="primary", on_click=delete_selected_windows_callback)

        elif admin_mode == "User Management":
            st.subheader("Manage Users")

            with st.form("add_user_form", clear_on_submit=True):
                st.write("**Create a New User**")
                new_username = st.text_input("New Username")
                new_password = st.text_input("New Password", type="password")
                new_password_confirm = st.text_input("Confirm New Password", type="password")
                new_role = st.selectbox("User Role", ["viewer", "admin"])

                submitted = st.form_submit_button("Create User")
                if submitted:
                    if not all([new_username, new_password, new_password_confirm, new_role]):
                        st.warning("All fields are required.")
                    elif new_password != new_password_confirm:
                        st.error("Passwords do not match.")
                    else:
                        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                        result = database.add_user(new_username, hashed_password, new_role)
                        if result is True:
                            st.success(f"User '{new_username}' created successfully!")
                            st.rerun() 
                        else:
                            st.error(f"Failed to create user: {result}")

            st.divider()
            st.write("**Existing Users**")
            all_users_df = database.get_all_users()

            if not all_users_df.empty:
                users_to_display = all_users_df[all_users_df['username'] != st.session_state.admin_user]
                st.dataframe(users_to_display[['username', 'role']], use_container_width=True)

                if not users_to_display.empty:
                    with st.expander("Delete Users"):
                        users_to_delete = st.multiselect(
                            "Select users to delete:",
                            options=users_to_display['id'],
                            format_func=lambda user_id: users_to_display[users_to_display['id'] == user_id]['username'].iloc[0]
                        )

                        if st.button("Delete Selected Users", type="primary"):
                            if not users_to_delete:
                                st.warning("Please select at least one user to delete.")
                            else:
                                deleted_count = 0
                                for user_id in users_to_delete:
                                    if database.delete_user(user_id):
                                        deleted_count += 1
                                st.success(f"Successfully deleted {deleted_count} user(s).")
                                st.rerun() 
            else:
                st.info("No other users found.")

        st.divider()
        st.button("Logout", on_click=logout_callback)