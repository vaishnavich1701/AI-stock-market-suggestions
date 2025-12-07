from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

print("Connecting to:", os.getenv("MONGO_DB_URI"))

try:
    client = MongoClient(os.getenv("MONGO_DB_URI"), serverSelectionTimeoutMS=5000)
    client.server_info()
    print("🎉 Connected to MongoDB Atlas!")
    print("Databases:", client.list_database_names())
except Exception as e:
    print("❌ MongoDB Connection Error:", e)
