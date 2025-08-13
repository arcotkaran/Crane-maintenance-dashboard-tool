# FILE: auth.py

# SCRIPT NAME: auth.py
# DESCRIPTION: Handles user authentication by reading from the database.

import sqlite3
import bcrypt
from config import Paths, logger

def verify_user(username: str, password: str) -> str | None:
    """
    Verifies a user's credentials against the database.
    It fetches the hashed password for the given username and uses bcrypt to
    securely compare it with the provided password.

    Args:
        username (str): The username to check.
        password (str): The password to check.

    Returns:
        str: The username if the credentials are valid and the user is an 'admin'.
        None: If the credentials are invalid or the user is not an admin.
    """
    logger.info(f"Attempting to verify user: '{username}'")
    try:
        with sqlite3.connect(Paths.DATABASE_PATH) as conn:
            cursor = conn.cursor()
            # Find the user by their username
            cursor.execute("SELECT hashed_password, role FROM users WHERE username = ?", (username,))
            result = cursor.fetchone()

            if result:
                stored_hash, role = result
                # Securely check if the provided password matches the stored hash
                if bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
                    if role == 'admin':
                        logger.info(f"Verification successful for admin user: '{username}'")
                        return username  # Login successful
                    else:
                        logger.warning(f"User '{username}' attempted to log in but is not an admin.")
                        return None
    except sqlite3.Error as e:
        logger.error(f"Database error during user verification for '{username}': {e}")

    logger.warning(f"Verification failed for user: '{username}'")
    return None