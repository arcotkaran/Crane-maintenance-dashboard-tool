# SCRIPT NAME: auth.py
# DESCRIPTION: Handles user authentication by reading from an external users.json file.
# UPDATE: Simplified to only verify an admin password, without checking username.
# REFACTOR: Centralized path and logger configuration.

import json
from config import Paths, logger

USERS_DATA = {
  "users": [
    {
      "username": "user",
      "password": "password",
      "role": "viewer"
    },
    {
      "username": "manager",
      "password": "manager123",
      "role": "admin"
    },
    {
      "username": "admin",
      "password": "admin123",
      "role": "admin"
    }
  ]
}

def load_users():
    """
    Returns the hardcoded list of user objects.
    """
    logger.debug("Loading users from embedded data.")
    return USERS_DATA.get("users", [])

def verify_admin_password(password: str) -> str | None:
    """
    Checks if the provided password matches any user with the 'admin' role.

    It iterates through all users loaded from the JSON file and checks two conditions:
    1. The user's role is "admin".
    2. The user's password matches the one provided.

    Args:
        password (str): The password to check.

    Returns:
        str: The username of the matching admin if found.
        None: If no matching admin user is found.
    """
    logger.info("Verifying admin password.")
    all_users = load_users()
    for user in all_users:
        # Check if the user has the 'admin' role and the password matches.
        # .get() is used for safe access in case a key is missing from a user entry.
        if user.get("role") == "admin" and user.get("password") == password:
            username = user.get("username")
            logger.info(f"Password verification successful for admin user: '{username}'")
            return username # Return the admin's username on success
    
    logger.warning("Admin password verification failed. No matching user found.")
    return None # Return None if the loop finishes without finding a match
