from bson import ObjectId
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import traceback

load_dotenv()

try:
    print("Connecting to MongoDB...")
    client = MongoClient(os.getenv("MONGO_DB_URI"))
    db = client["ima-user"]

    print("Clearing existing data...")
    db.users.delete_many({})
    db.portfolios.delete_many({})
    db.portfolio_stocks.delete_many({})

    print("Inserting sample users...")
    users = [
        {
            "name": "Tech Focus Trader",
            "email": "techtrader@example.com",
            "questions": [
                {"question": "What is your risk appetite?", "answer": "High"},
                {"question": "Preferred investment horizon?", "answer": "Short Term"},
                {"question": "Any sector preference?", "answer": "Technology"}
            ]
        },
        {
            "name": "Safe Investor",
            "email": "safe@example.com",
            "questions": [
                {"question": "What is your risk appetite?", "answer": "Low"},
                {"question": "Preferred investment horizon?", "answer": "Long Term"},
                {"question": "Any sector preference?", "answer": "Healthcare"}
            ]
        }
    ]

    user_ids = db.users.insert_many(users).inserted_ids

    print("Users inserted:")
    for uid in user_ids:
        print("User ID:", str(uid))

    print("Inserting portfolios...")
    portfolios = [
        {"userId": user_ids[0], "name": "High Growth Tech Portfolio"},
        {"userId": user_ids[1], "name": "Safe & Stable Portfolio"}
    ]

    portfolio_ids = db.portfolios.insert_many(portfolios).inserted_ids

    print("Inserting portfolio stocks...")
    stocks = [
        {"portfolioId": portfolio_ids[0], "ticker": "AAPL", "sector": "Technology", "risk": "Medium"},
        {"portfolioId": portfolio_ids[0], "ticker": "GOOGL", "sector": "Technology", "risk": "Medium"},
        {"portfolioId": portfolio_ids[0], "ticker": "TSLA", "sector": "Automotive", "risk": "High"},
        {"portfolioId": portfolio_ids[1], "ticker": "JNJ", "sector": "Healthcare", "risk": "Low"},
        {"portfolioId": portfolio_ids[1], "ticker": "PG", "sector": "Consumer Staples", "risk": "Low"},
        {"portfolioId": portfolio_ids[1], "ticker": "KO", "sector": "Consumer Staples", "risk": "Low"},
    ]

    db.portfolio_stocks.insert_many(stocks)

    print("\n🎯 Sample data inserted successfully!")
    print("Tech User ID:   ", str(user_ids[0]))
    print("Stable User ID: ", str(user_ids[1]))

except Exception as e:
    print("\n❌ An error occurred while inserting data!")
    traceback.print_exc()
