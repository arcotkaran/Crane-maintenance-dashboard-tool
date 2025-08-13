# SCRIPT NAME: config.py
# DESCRIPTION: A centralized configuration file for all paths, settings, and logging.
# UPDATE: Modified logger setup to be more robust for the Streamlit execution environment.
# UPDATE 2: Added TAG_IDS_FILE and TAG_CHANGE_FILE to centralize all path configurations.

import os
import logging
import sys

class Paths:
    """A class to hold all static paths for the application."""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # --- File and Directory Paths ---
    LOG_DIRECTORY = os.path.join(BASE_DIR, "logs")
    DATABASE_PATH = os.path.join(BASE_DIR, "crane_data.db")
    USERS_FILE_PATH = os.path.join(BASE_DIR, "users.json")
    DEBUG_LOG_FILE_PATH = os.path.join(LOG_DIRECTORY, "debug.txt")
    SERVICE_CONFIG_FILE = os.path.join(BASE_DIR, "service_config.csv")
    FLEET_MAP_FILE = os.path.join(BASE_DIR, "fleet_map.csv")

    # --- Paths for CraneSats-Zip_db script ---
    TAG_IDS_FILE = os.path.join(BASE_DIR, "tag_ids.txt")
    TAG_CHANGE_FILE = os.path.join(BASE_DIR, "tag_change.csv")


    @staticmethod
    def ensure_log_directory_exists():
        """Creates the log directory if it doesn't exist."""
        if not os.path.exists(Paths.LOG_DIRECTORY):
            os.makedirs(Paths.LOG_DIRECTORY)
            print(f"Created directory: {Paths.LOG_DIRECTORY}")

class Settings:
    """A class to hold all application-wide settings and configurable variables."""
    # --- Dashboard Settings ---
    DATE_FORMAT = "%d/%m/%y"
    CONFLICT_WINDOW_MINUTES = 90
    
    # --- Database Table Names ---
    STATS_TABLE_NAME = "crane_stats"
    SERVICE_LOG_TABLE_NAME = "service_log"
    MAINTENANCE_WINDOWS_TABLE_NAME = "maintenance_windows"
    PREDICTIONS_TABLE_NAME = "predictions"

class Logging:
    """A class to configure and provide a logger instance."""
    @staticmethod
    def setup_logger():
        """
        Configures and returns a shared logger instance.
        This method is designed to be robust for Streamlit's re-execution model by clearing
        existing handlers before adding new ones, preventing duplicate log entries.
        """
        logger = logging.getLogger("crane_dashboard")
        logger.setLevel(logging.DEBUG)

        # Clear existing handlers to prevent duplicate logs in Streamlit's execution environment
        if logger.hasHandlers():
            logger.handlers.clear()

        # File Handler for detailed debug logging
        try:
            # Ensure the log directory exists before trying to create the file handler.
            Paths.ensure_log_directory_exists()
            file_handler = logging.FileHandler(Paths.DEBUG_LOG_FILE_PATH, mode='w', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)-8s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # Use print for critical errors in case logger itself fails
            print(f"CRITICAL: Could not set up file logger. Error: {e}")

        # Console Handler for higher-level info
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_formatter = logging.Formatter('%(levelname)-8s - %(module)s - %(message)s')
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)
        
        logger.info("Logging configured successfully.")
        
        return logger

# --- Initialize Logger ---
# This setup is now robust for re-execution in Streamlit.
logger = Logging.setup_logger()
