"""
manage_users.py
-----------------
Small command-line helper for adding user accounts to the dashboard's
login system, beyond the single bootstrap admin account.

Usage:
    python manage_users.py add <username> <role>
        (you'll be prompted for a password so it never appears in your
         shell history or in a screen recording)

Example:
    python manage_users.py add nurse1 clinician
"""

import getpass
import sys

import db


def main():
    if len(sys.argv) != 4 or sys.argv[1] != "add":
        print("Usage: python manage_users.py add <username> <role>")
        print("Example: python manage_users.py add nurse1 clinician")
        sys.exit(1)

    _, _, username, role = sys.argv

    password = getpass.getpass(f"Set a password for '{username}': ")
    confirm = getpass.getpass("Confirm password: ")

    if password != confirm:
        print("Passwords did not match. No user was created.")
        sys.exit(1)

    if len(password) < 8:
        print("Please use a password of at least 8 characters.")
        sys.exit(1)

    db.init_db()
    created = db.create_user(username, password, role=role)

    if created:
        print(f"User '{username}' created with role '{role}'.")
    else:
        print(f"A user named '{username}' already exists. No changes made.")


if __name__ == "__main__":
    main()
