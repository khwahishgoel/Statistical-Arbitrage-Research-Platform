"""Quick Atlas connection test."""
from pathlib import Path
from dotenv import load_dotenv

# Explicitly load .env from project root (works regardless of where you run from)
load_dotenv(Path(__file__).parent / ".env")

from pymongo import MongoClient
import os

uri = os.getenv("MONGO_URI")

if not uri:
    print("ERROR: MONGO_URI not found.")
    print(f"  Looking for .env in: {Path(__file__).parent}")
    print("  Make sure .env is in the same folder as this script.")
    exit(1)

print(f"Connecting to: {uri[:50]}...")

client = MongoClient(uri, serverSelectionTimeoutMS=5000)
info = client.server_info()
print(f"Connected! MongoDB version: {info['version']}")

db = client["stat_arb"]
db["connection_test"].insert_one({"status": "ok"})
db["connection_test"].drop()
print("Read/write test: PASS")
print("\nAll good — run phase1_data_pipeline.py next.")
client.close()