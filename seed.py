"""
seed.py — Create a test API key for development
─────────────────────────────────────────────────
Run ONCE after starting the server:
    python seed.py

This creates a test API key in the database.
Use this key in all your test requests:
    X-API-Key: test_key_12345
"""

from database import create_tables, seed_test_api_key

if __name__ == "__main__":
    print("Creating tables...")
    create_tables()
    print("Seeding test API key...")
    seed_test_api_key()
    print("\nDone! Use this header in all requests:")
    print("  X-API-Key: test_key_12345")