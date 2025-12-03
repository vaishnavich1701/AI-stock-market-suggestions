import pymongo
from bson.objectid import ObjectId
from datetime import datetime

# Atlas Connection URI
MONGO_URI = "mongodb+srv://ima_admin:MyDBPass2025!@ima-stock-ai.xxxxxx.mongodb.net/?retryWrites=true&w=majority"

DB_NAME = "ima-user"


def setup_database():
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client[DB_NAME]

        print("Connected to Atlas!")

        # Collections
        users_col = db["users"]
        portfolios_col = db["portfolios"]
        portfolio_stocks_col = db["portfolio_stocks"]

        # Indexes
        portfolios_col.create_index("userId")
        portfolio_stocks_col.create_index("portfolioId")
        portfolio_stocks_col.create_index("ticker")

        print("Collections + Indexes Ready!")

        # Sample Data
        user_id = users_col.insert_one({
            "name": "Demo User",
            "email": "demo@example.com",
            "createdAt": datetime.utcnow(),
            "questions": [
                {"question": "Risk Appetite?", "answer": "Medium"},
                {"question": "Preferred Duration?", "answer": "Short-term"}
            ]
        }).inserted_id

        portfolio_id = portfolios_col.insert_one({
            "userId": ObjectId(user_id),
            "name": "Starter Portfolio",
            "createdAt": datetime.utcnow()
        }).inserted_id

        portfolio_stocks_col.insert_many([
            {"portfolioId": portfolio_id, "ticker": "AAPL"},
            {"portfolioId": portfolio_id, "ticker": "GOOGL"},
            {"portfolioId": portfolio_id, "ticker": "TSLA"}
        ])

        print("Sample data inserted successfully!")
        print("🎉 MongoDB Atlas Setup Completed!")

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    setup_database()
