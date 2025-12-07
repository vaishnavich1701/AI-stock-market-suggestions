from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from bson import ObjectId
from typing import List
import pymongo
import requests
import openai
import os
from dotenv import load_dotenv
import json
import uvicorn
from datetime import datetime, timedelta
from bson.errors import InvalidId
from fastapi.responses import JSONResponse
import time 



# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Fetch credentials from environment variables
OPENAI_API_KEY = os.getenv("")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# MongoDB Connection
MONGO_DB_URI = os.getenv("MONGO_DB_URI")
mongo_client = pymongo.MongoClient(MONGO_DB_URI)


# Connect to the ima-user database
db = mongo_client["ima-user"]
users_collection = db["users"]
portfolios_collection = db["portfolios"]
portfolio_stocks_collection = db["portfolio_stocks"]


def fetch_current_price(ticker: str):
    url = f"https://api.polygon.io/v2/last/trade/{ticker.upper()}?apiKey={POLYGON_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'results' in data and 'p' in data['results']:
            return data['results']['p']
    return None


# Helper function to validate ObjectId
def get_valid_objectid(user_id: str):
    try:
        return ObjectId(user_id)
    except Exception:
        raise ValueError("Invalid user ID format. Please provide a valid ObjectId.")


# Helper functions for MongoDB queries
def fetch_user_data(user_id: ObjectId):
    user = users_collection.find_one({"_id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return user


def fetch_portfolios(user_id: ObjectId):
    return list(portfolios_collection.find({"userId": user_id}))


def fetch_portfolio_stocks(portfolio_id: ObjectId):
    x=list(portfolio_stocks_collection.find({"portfolioId": portfolio_id}))
    print(x)
    return list(portfolio_stocks_collection.find({"portfolioId": portfolio_id}))


def fetch_company_data(ticker: str):
    url = f"https://api.polygon.io/v3/reference/tickers/{ticker.upper()}?apiKey={POLYGON_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return {}


def fetch_stock_trends(ticker: str):
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=90)  # Fetch data for the last 90 days
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?apiKey={POLYGON_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("results", [])
    return []


def analyze_risk(stock_trends: List[dict], stock_info: dict):
    if not stock_trends:
        return "Unknown"
    price_changes = [abs(trend["c"] - trend["o"]) for trend in stock_trends]
    avg_change = sum(price_changes) / len(price_changes)
    beta = stock_info.get("beta", "N/A")
    pe_ratio = stock_info.get("peRatio", "N/A")
    risk_level = "Low Risk"
    if avg_change > 5 or (beta != "N/A" and beta > 1.2) or (pe_ratio != "N/A" and pe_ratio > 25):
        risk_level = "High Risk"
    elif avg_change > 2 or (beta != "N/A" and beta > 1.0):
        risk_level = "Medium Risk"
    return risk_level


def get_historical_data_summary(stocks):
    historical_data_summary = ""
    for stock in stocks:
        ticker = stock.get("ticker")
        stock_info = fetch_company_data(ticker)
        stock_trends = fetch_stock_trends(ticker)
        current_price = fetch_current_price(ticker)
        risk = analyze_risk(stock_trends, stock_info)
        stock["sector"] = stock_info.get("sector")
        stock["risk"] = risk
        stock["current_price"] = current_price
        historical_data_summary += f"\nTicker: {ticker}\nRisk: {risk}\nCurrent Price: {current_price}\n"
        historical_data_summary += "\n".join(
            [f"- Date: {datetime.fromtimestamp(trend['t']/1000).strftime('%Y-%m-%d')}, Open: {trend['o']}, Close: {trend['c']}" for trend in stock_trends[:5]]
        )
    return historical_data_summary


def extract_percentage(percentage_str):
    import re
    match = re.search(r"([\+\-]?[\d\.]+)%", percentage_str)
    if match:
        return float(match.group(1))
    return None

def extract_numeric_value(price_str):
    import re
    matches = re.findall(r"[\d\.]+", price_str.replace(',', ''))
    if matches:
        return float(matches[0])
    return None

def calculate_potential_upside(ticker: str):
    stock_trends = fetch_stock_trends(ticker)
    if not stock_trends:
        return None
    # Calculate average daily percentage change over the last 90 days
    daily_changes = [
        (trend['c'] - trend['o']) / trend['o'] * 100
        for trend in stock_trends if trend['o'] != 0
    ]
    if not daily_changes:
        return None
    average_daily_change = sum(daily_changes) / len(daily_changes)
    # Limit the potential upside to a realistic range
    MAX_DAILY_MOVEMENT_PERCENT = 5.0
    potential_upside = min(max(average_daily_change, -MAX_DAILY_MOVEMENT_PERCENT), MAX_DAILY_MOVEMENT_PERCENT)
    return potential_upside

def generate_suggestions_with_gpt(portfolio_name: str, stocks: List[dict], user_preferences: List[dict], historical_data_summary: str):
    stock_list = "\n".join([
        f"{stock['ticker']} - Sector: {stock.get('sector', 'Unknown')}, Risk: {stock.get('risk')}, Current Price: {stock.get('current_price')}"
        for stock in stocks
    ])
    preferences_text = "\n".join([f"{q['question']}: {q['answer']}" for q in user_preferences])
    prompt = f"""
A user is actively trading with a portfolio named '{portfolio_name}', currently holding the following positions:
{stock_list}

The historical performance and risk assessment of the current positions are summarized as follows:
{historical_data_summary}

The user has shared the following investment preferences and goals:
{preferences_text}

As an expert stock market analyst, your task is to recommend 3 additional stocks or trading strategies that align with the user's goals and improve the portfolio's overall performance.
Also include a section with predictions for the user's current stocks as well which are present in their portfolio.

To provide effective recommendations, you must consider the following cardinal trading principles:

1. **Diversification**: Avoid over-concentration in a single stock, sector, or strategy to minimize systemic risks.
2. **Risk Management**: Suggest positions with a balanced risk-reward profile, taking into account historical volatility and appropriate stop-loss levels.
3. **Market Trends**: Incorporate recent market data, including earnings reports, sector trends, macroeconomic indicators (e.g., GDP, employment rates), and geopolitical developments.
4. **Liquidity and Transaction Costs**: Recommend trades that offer high liquidity and tight spreads to optimize execution costs.
5. **Time Horizon Alignment**: Ensure the proposed trades align with the user's preferred trading horizon, whether short-term, medium-term, or long-term.

###
For each recommendation:
1. Explain how the recommendation aligns with the cardinal principles mentioned above.
2. Justify the choice based on current market trends, economic indicators, and relevant industry or stock-specific developments.
3. Highlight how the recommendation improves the portfolio's diversification and reduces overall risk exposure.
4. Evaluate the potential for growth, stability, or income generation and its impact on the portfolio's risk-return profile.
###

'''
Provide your recommendations in the format below:
1. Stock Ticker (Sector, Risk)
2. Recommendation Rationale (aligning with cardinal principles, market trends, and risk management)
3. Potential Growth, Stability, or Income Potential (if applicable)
4. Historical Analysis (e.g., past performance, risk assessment)
5. Why to consider the recommendation
6. Prediction for current stocks in your portfolio for the next 24 hours.
'''

---
Important points to consider:
1. Potential Upside Integration: Include a clear potential upside percentage for the next 24 hours for each stock or suggestion, explaining its significance to the decision-making process.

2. Data-Driven Confidence: Back up recommendations with concrete numbers and data, such as historical price trends, recent performance, and potential future gains.

3. Humanized Tone: Use a conversational tone rather than a robotic one, making it relatable and easy to understand for users.

4. Reason to Invest: Explain why this stock or suggestion is worth considering. Highlight factors like market position, growth potential, or industry trends.

5. Historical Context: Provide historical price data and trends to show how the stock has performed over time and why it has potential for growth. It should be well defined and the terminology should be specific to what is used in stock market. Be precise and to the point.

6. Stock Market Jargon: Use appropriate terminologies that a stock market analyst would typically employ, such as P/E ratios, market cap, dividend yield, etc., to build credibility.

7. Explain the "Why": Add reasoning for the suggestion, such as strong fundamentals, recent positive news, or alignment with market trends.

8. Potential Projections: Offer projections on what the stock's price could be in the near or long term and support this with logical arguments.

9. Build Trust: Make the suggestions look trustworthy by combining data, logical reasoning, and an approachable tone.

10. Summary for Decision Making: Conclude with a summary that helps users quickly understand why they should consider the suggestion.

11. Always consider the current price of a stock which is being given to you in the {stock_list}.
-----

____

Provide your recommendations and predictions in the following JSON format:
{{
  "suggestions": [
    {{
      "ticker": "AAPL",
      "sector": "Technology",
      "risk": "Medium Risk",
      "rationale": "AAPL adds large-cap tech exposure, balancing mid-cap stocks in the portfolio. Apple's expansion into services and wearables drives optimism in the tech sector. With a beta of 1.2, AAPL offers moderate risk suitable for a balanced portfolio.",
      "growthPotential": "Analysts expect a 10% YoY revenue increase, supported by strong product demand.",
      "historicalAnalysis": "AAPL has consistently outperformed the S&P 500 over the past decade. Stable performance during market corrections highlights its defensive qualities.",
      "potentialUpside": "15%"
    }},
    {{
      "ticker": "MSFT",
      "sector": "Technology",
      "risk": "Low Risk",
      "rationale": "MSFT provides strong diversification through its enterprise and cloud offerings. The recent Azure growth figures highlight Microsoft's continued market leadership. With a beta of 0.9, MSFT is a low-risk addition to any portfolio.",
      "growthPotential": "Continued revenue growth in the cloud segment, with an expected increase of 12% in the next fiscal year.",
      "historicalAnalysis": "MSFT has shown resilience during market downturns, maintaining steady growth over the past five years.",
      "potentialUpside": "10%"
    }},
    {{
      "ticker": "JNJ",
      "sector": "Healthcare",
      "risk": "Low Risk",
      "rationale": "JNJ provides exposure to the defensive healthcare sector, which tends to perform well in volatile markets. With strong fundamentals and low volatility, JNJ offers a stable investment option.",
      "growthPotential": "With a strong pipeline of drugs and increasing global demand for healthcare products, JNJ is positioned for steady growth.",
      "historicalAnalysis": "JNJ has maintained consistent dividend payouts and has a track record of outperforming during economic slowdowns.",
      "potentialUpside": "8%"
    }}
  ],
  "predictions_for_24h": [
    {{
      "ticker": "TSLA",
      "confidence": "High",
      "potentialChange": "+2.9%"
    }},
    {{
      "ticker": "AMZN",
      "confidence": "Medium",
      "potentialChange": "+1.5%"
    }},
    {{
      "ticker": "GOOGL",
      "confidence": "High",
      "potentialChange": "+2.0%"
    }}
  ]
}}

- Ensure the output is valid JSON and does not include extraneous symbols or text.
____

////////////////////

### Important Notes:
- Be concise but comprehensive in each section.
- Provide all required sections for each stock recommendation.
- Use a uniform tone and terminology across all recommendations.
- DO NOT generate anything else apart from the actual output.
- Include more technical terminologies of the stock market in the output and make it sound professional.
- Be very precise and to the point.
- DO NOT include any markdown symbols like #, *, etc.
- Add a prediction, based on the historical data and current market trends for each stock which is present in the user portfolio. The prediction should be for the next 24 hours.
- Provide the output ONLY in valid JSON format with proper indentation.
- The output should not have any leading or trailing ''' or json''' symbols.
- Use the current prices provided when calculating target prices and potential upside or downside percentages.
- **In your output, for each suggestion, include a field "targetPrice_for_24h" to represent the target price for the next 24 hours, calculated based on the current price and potential upside percentage.**
- **Keep the potential upside and downside VERY realistic and according to the current market conditions and current prices.**
- **In your output, for each suggestion, include a field "targetPrice_for_24h" to represent the target price for the next 24 hours, calculated based on the current price and potential upside percentage. Do not include "predictedPrice_for_24h" in the suggestions.**

- **In your "predictions_for_24h" section, include the field "predictedPrice_for_24h" to represent the predicted price for the next 24 hours.**

- **Keep the potential upside and downside VERY realistic and according to the current market conditions and current prices.**

Provide your recommendations in this exact format for consistency. ONLY GIVE VALID JSON IN THE FORMAT SPECIFIED. DO NOT ADD ANY LEADING OR TRAILING ''' OR JSON SYMBOLS OR '''json in the output.

////////////////////


Your recommendations should be detailed, actionable, and presented in a manner that the user can directly apply. Ensure each recommendation is well-supported with logical and data-driven reasoning.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-5.1",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.0
        )    
        output = response.choices[0].message.content.strip()
        print("GPT Response: ", output)
        suggestions_output = json.loads(output)
    
        # Process suggestions to add currentPrice, targetPrice_for_24h, and targetPercent
        suggestions = suggestions_output.get("suggestions", [])
        for suggestion in suggestions:
            ticker = suggestion.get("ticker")
            # Use the potential upside calculated earlier
            potential_upside_percent = calculate_potential_upside(ticker)
            # Fetch current price
            current_price = fetch_current_price(ticker)
            if current_price is not None and potential_upside_percent is not None:
                # Compute targetPrice_for_24h
                target_price = current_price * (1 + potential_upside_percent / 100)
                suggestion["currentPrice"] = f"{current_price:.2f}"
                suggestion["targetPrice_for_24h"] = f"{target_price:.2f}"
                # Remove 'predictedPrice_for_24h' if present
                if 'predictedPrice_for_24h' in suggestion:
                    del suggestion['predictedPrice_for_24h']
                # Format targetPercent with '+' or '-' sign
                if potential_upside_percent >= 0:
                    target_percent_str = f"+{potential_upside_percent:.2f}"
                else:
                    target_percent_str = f"{potential_upside_percent:.2f}"
                suggestion["targetPercent"] = target_percent_str
                # Include potentialUpside
                suggestion["potentialUpside"] = f"{potential_upside_percent:.2f}%"
            else:
                # Handle missing data
                suggestion["currentPrice"] = None
                suggestion["targetPrice_for_24h"] = None
                suggestion["targetPercent"] = None
                suggestion["potentialUpside"] = None
                # Remove 'predictedPrice_for_24h' if present
                if 'predictedPrice_for_24h' in suggestion:
                    del suggestion['predictedPrice_for_24h']
    
        # Similarly process predictions
        predictions = suggestions_output.get("predictions_for_24h", [])
        for prediction in predictions:
            ticker = prediction.get("ticker")
            potential_change_percent = calculate_potential_upside(ticker)
            current_price = fetch_current_price(ticker)
            if current_price is not None and potential_change_percent is not None:
                predicted_price = current_price * (1 + potential_change_percent / 100)
                prediction["currentPrice"] = f"{current_price:.2f}"
                prediction["predictedPrice_for_24h"] = f"{predicted_price:.2f}"
                if potential_change_percent >= 0:
                    percentage_str = f"+{potential_change_percent:.2f}"
                else:
                    percentage_str = f"{potential_change_percent:.2f}"
                prediction["percentage"] = percentage_str
                prediction["potentialChange"] = percentage_str + "%"
            else:
                prediction["currentPrice"] = None
                prediction["predictedPrice_for_24h"] = None
                prediction["percentage"] = None
                prediction["potentialChange"] = None
            # Remove 'targetPrice_for_24h' if present
            if 'targetPrice_for_24h' in prediction:
                del prediction['targetPrice_for_24h']
    
        return suggestions_output
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Error parsing GPT response: {e}")
    except Exception as e:
     print("❌ Full Error Details:", e)
    import traceback
    traceback.print_exc()
    raise HTTPException(status_code=500, detail=f"Error generating suggestions: {str(e)}")



# API Endpoint to fetch portfolio suggestions
@app.get("/portfolio/suggestions/")
def get_portfolio_suggestions(user_id: str):
    
    start_time = time.time()  
    try:
        user_id_obj = get_valid_objectid(user_id)
        user = fetch_user_data(user_id_obj)
        portfolios = fetch_portfolios(user_id_obj)
        if not portfolios:
            raise HTTPException(status_code=404, detail="No portfolios found for the user.")
        response_data = []
        for portfolio in portfolios:
            portfolio_id = portfolio["_id"]
            stocks = fetch_portfolio_stocks(portfolio_id)
            if not stocks:
                continue
            # Generate historical data summary and update stocks
            historical_data_summary = get_historical_data_summary(stocks)
            # Prepare stock data for GPT
            stock_data = [{
                "ticker": stock.get("ticker"),
                "sector": stock.get("sector"),
                "risk": stock.get("risk"),
                "current_price": stock.get("current_price")
            } for stock in stocks]
            suggestions = generate_suggestions_with_gpt(
                portfolio["name"], stock_data, user.get("questions", []), historical_data_summary
            )
            response_data.append({
                "portfolio": portfolio["name"],
                "suggestions": suggestions["suggestions"]
            })
        end_time = time.time() - start_time  
        print(f"Final portfoilo Suggestions response time: {end_time:.2f} seconds")
  
        return {"data": response_data}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/portfolio/predictions/")
def get_portfolio_predictions(user_id: str):
    start_time = time.time()
    try:
        user_id_obj = get_valid_objectid(user_id)
        user = fetch_user_data(user_id_obj)
        portfolios = fetch_portfolios(user_id_obj)
        if not portfolios:
            raise HTTPException(status_code=404, detail="No portfolios found for the user.")
        response_data = []
        for portfolio in portfolios:
            portfolio_id = portfolio["_id"]
            stocks = fetch_portfolio_stocks(portfolio_id)
            if not stocks:
                continue
            # Generate historical data summary and update stocks
            historical_data_summary = get_historical_data_summary(stocks)
            # Prepare stock data for GPT
            stock_data = [{
                "ticker": stock.get("ticker"),
                "sector": stock.get("sector"),
                "risk": stock.get("risk"),
                "current_price": stock.get("current_price")
            } for stock in stocks]
            suggestions = generate_suggestions_with_gpt(
                portfolio["name"], stock_data, user.get("questions", []), historical_data_summary
            )
            response_data.append({
                "portfolio": portfolio["name"],
                "predictions_for_24h": suggestions["predictions_for_24h"]
            })
        end_time = time.time() - start_time  
        print(f"Final Portfolio Predictions  response time: {end_time:.2f} seconds")
        return {"data": response_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class RecommendationRequest(BaseModel):
    user_id: str

db = mongo_client['ima-user']
collection = db['users']

def get_user_by_id2(user_id: str):
    """Fetch user information from MongoDB using the user's _id."""
    try:

        user = collection.find_one({"_id": ObjectId(user_id)})
        if user:
            return user 
        else:
            print("User not found.")
            return None
    except Exception as e:
        print(f"Error fetching user by ID: {e}")
        return None


def get_user_questions(user_id):
    """Fetch user-specific questions and answers from MongoDB by _id."""
    user = get_user_by_id2(user_id)  
    if user and 'questions' in user:
        structured_data = [{"question": q["question"], "answer": q["answer"]} for q in user['questions']]
        # print("User Questions",structured_data)
        return structured_data
    return None


def fetch_current_price(ticker: str):
    url = f"https://api.polygon.io/v2/last/trade/{ticker.upper()}?apiKey={POLYGON_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'results' in data and 'p' in data['results']:
            return data['results']['p'] 
    return None


def fetch_tickers_by_market(market_type="stocks", locale="us"):
    """Fetch a list of tickers dynamically from Polygon API based on market type and locale."""
    url = f"https://api.polygon.io/v3/reference/tickers?market={market_type}&locale={locale}&apiKey={POLYGON_API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if 'results' in data:
            # Return essential ticker data
            return [
                {
                    "ticker": stock["ticker"],
                    "name": stock.get("name", "N/A"),
                    "market": stock.get("market", "N/A"),
                    "locale": stock.get("locale", "N/A")
                }
                for stock in data['results']
            ] 
    print(f"Error or no data found: {response.status_code}")
    return []


def analyze_sentiments(user_preferences):
    """Analyze sentiment of user preferences using gpt-4o."""
    prompt = f"""
    You are an AI assistant analyzing user investment preferences. Based on the following user responses:
    
    {json.dumps(user_preferences, indent=2)}

    For each question, provide a sentiment analysis result (Positive, Neutral, Negative) and classify the user's preferences:
    - High Risk: True or False
    - Short-Term Focus: True or False
    - Income Focus: True or False
    - Long-Term Focus: True or False
    - Sector Preference: Technology, Healthcare, Finance, or Other
    - Volatility Tolerance: High, Medium, Low

    Return the result in JSON format, with sentiment for each question.
    """  
    response = openai.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": "You are an expert in analyzing user sentiments for investment preferences."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
       
    sentiments_analysis = response.choices[0].message.content
    # print("sentimental analysis data: ", sentiments_analysis)
    try:
        sentiments_data = json.loads(sentiments_analysis)
        print("Sentiments Analysis:", sentiments_data)
        return sentiments_data
    except json.JSONDecodeError:
        # print("Error: Sentiment analysis response is not valid JSON.")
        return None


def analyze_risk2(ticker):
    """Analyze stock risk using Polygon API."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/2024-01-01/2024-10-31?adjusted=true&sort=asc&apiKey={POLYGON_API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if "results" in data and data["results"]:
            high_prices = [entry["h"] for entry in data["results"]]
            low_prices = [entry["l"] for entry in data["results"]]

            volatility = max(high_prices) - min(low_prices)
            trend = "Going Up" if high_prices[-1] > low_prices[0] else "Going Down"

            return {
                "volatility": round(volatility, 2),
                "trend": trend
            }

        else:
            return {"volatility": None, "trend": "Unknown"}
    else:
        return {"volatility": None, "trend": "Unknown"}

def fetch_historical_data(ticker, start_date, end_date):
    """Fetch historical ticker data from the Polygon API."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/month/{start_date}/{end_date}?adjusted=true&sort=asc&apiKey={POLYGON_API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if "results" in data and data["results"]:
            return data["results"]
        else:
            # print(f"No historical data found for {ticker}.")
            return None
    else:
        print(f"Error fetching historical data for {ticker}: {response.status_code}")
        return None

def analyze_historical_data(historical_data):
    """Analyze historical data to derive insights."""
    if not historical_data:
        return "Historical data not available."

    prices = [entry["vw"] for entry in historical_data]
    high_prices = [entry["h"] for entry in historical_data]
    low_prices = [entry["l"] for entry in historical_data]

    avg_price = sum(prices) / len(prices)
    volatility = max(high_prices) - min(low_prices)

    trend = "Going Up" if prices[-1] > prices[0] else "Going Down"

    return f"""
    - Average Price: ${avg_price:.2f}
    - Volatility: ${volatility:.2f}
    - Trend: {trend}
    """
def analyze_trend(historical_data):
    """
    Provide detailed trend analysis with month-by-month percentage changes.
    
    Args:
    historical_data (list): Historical price data from Polygon API
    
    Returns:
    dict: Trend analysis with specific monthly performance insights
    """
    if not historical_data or len(historical_data) < 2:
        return {
            "trend_description": "Insufficient data for trend analysis",
            "monthly_trends": []
        }
    
    # month-to-month percentage changes
    monthly_trends = []
    for i in range(1, len(historical_data)):
        prev_price = historical_data[i-1]['vw']
        current_price = historical_data[i]['vw']
        
        percent_change = ((current_price - prev_price) / prev_price) * 100
        monthly_trends.append({
            "month": i,
            "percent_change": round(percent_change, 2)
        })
    
    # Aggregate 
    avg_monthly_change = sum(trend['percent_change'] for trend in monthly_trends) / len(monthly_trends)
    
    #  overall trend description
    if avg_monthly_change > 5:
        trend_description = f"Strong upward trend with average monthly growth of {round(avg_monthly_change, 2)}%"
    elif avg_monthly_change > 0:
        trend_description = f"Moderate upward trend, averaging {round(avg_monthly_change, 2)}% monthly increase"
    elif avg_monthly_change == 0:
        trend_description = "Stable with minimal price fluctuations"
    elif avg_monthly_change > -5:
        trend_description = f"Moderate downward trend, averaging {round(abs(avg_monthly_change), 2)}% monthly decline"
    else:
        trend_description = f"Significant downward trend with average monthly decline of {round(abs(avg_monthly_change), 2)}%"
    
    return {
        "trend_description": trend_description,
        "monthly_trends": monthly_trends,
        "best_month": max(monthly_trends, key=lambda x: x['percent_change']),
        "worst_month": min(monthly_trends, key=lambda x: x['percent_change'])
    }


def get_portfolio_stocks(user_id: str):
    """Fetch all stocks across all portfolios for the user."""
    try:
        user_id = ObjectId(user_id)  
        portfolios = fetch_portfolios(user_id)  

        portfolio_stocks = []
        for portfolio in portfolios:
            portfolio_id = portfolio["_id"]  # 
            stocks = fetch_portfolio_stocks(portfolio_id)  
            portfolio_stocks.extend([stock["ticker"].strip().upper() for stock in stocks])  

        # print(f"Existing stocks for user {user_id}: {portfolio_stocks}")
        return portfolio_stocks
    except Exception as e:
        print(f"Error fetching portfolio stocks for user {user_id}: {e}")
        return []



def fetch_final_recommendations(sentiments_analysis, tickers_data,user_id):
    """Filter and recommend stocks based on user sentiment and risk analysis with an added LLM layer."""

    existing_stocks = get_portfolio_stocks(user_id)
    # print(f"Existing stocks for user {user_id}: {existing_stocks}")
    
    recommendations = []

    for ticker_data in tickers_data[:0]: 
        ticker = ticker_data["ticker"].strip().upper()
        
        # Skip stocks already in the portfolio
        if ticker in existing_stocks:
            print(f"Skipping existing stock: {ticker}")
            continue
        name = ticker_data.get("name", "Unknown")
        sector = ticker_data.get("market", "Unknown")

        # Fetch historical and risk analysis
        historical_data = fetch_historical_data(ticker, "2024-01-01", "2024-12-31")  
        trend_analysis = analyze_trend(historical_data)
        historical_analysis = analyze_historical_data(historical_data)    
        risk_data = analyze_risk2(ticker)
        current_price = fetch_current_price(ticker)

        recommendations.append({
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "historical_analysis": historical_analysis,
            "trend_analysis": trend_analysis,
            "risk_analysis": risk_data,
            "current_price": current_price
        })

    # Refine recommendations based on sentiment analysis
    refined_recommendations = [
        rec for rec in recommendations if rec["risk_analysis"]["volatility"] and rec["risk_analysis"]["volatility"] < 20
    ]

    # LLM layer for final enhancement
    llm_prompt = f"""
    As an expert stock market analyst, refine the stock recommendations to align with user preferences and market conditions.
 
    ---
    **Rules for Recommendations:**

    1. **Diversification:**
    - Avoid over-concentration in any  single stock, sector, or strategy.

    2. **Risk Management:**
    - Evaluate the risk-reward user preferences of each suggestion.
    - Consider historical volatility and establish clear stop-loss levels.

    3. **Market Trends:**
    - Integrate insights from recent market data such as earnings reports, sector trends, macroeconomic indicators (e.g., GDP, employment rates), and geopolitical developments.

    4. **Liquidity and Transaction Costs:**
    - Propose trades with high liquidity to ensure seamless execution.
    - Minimize transaction costs by focusing on options with tight bid-ask spreads.

    5. **Time Horizon Alignment:**
    - Match stock recommendations with the user’s specified trading horizon:
        - **Short-Term:** Focus on momentum and high-volatility stocks.
        - **Medium-Term:** Emphasize growth potential and solid fundamentals.
        - **Long-Term:** Prioritize stability, dividends, and blue-chip stocks.

    **Execution Guidelines:**
    - Always justify recommendations with data-backed reasoning.
    - Ensure that your output is tailored to the user's prefences , specified risk appetite and investment objectives.

    ---


    User Sentiment Analysis:
    {json.dumps(sentiments_analysis, indent=2)}

    Initial Recommendations:
    {json.dumps(refined_recommendations, indent=2)}

    Please suggest three refined recommendations in JSON format with detailed reasoning.
    """ 
    response = openai.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": "You are an expert financial advisor."},
            {"role": "user", "content": llm_prompt}
        ],
        temperature=1.0
    )
  
    try:
        final_recommendations = json.loads(response.choices[0].message.content)
        print("Refined Recommendations by LLM:", final_recommendations)
        return final_recommendations
    except json.JSONDecodeError:
        # print("Error: LLM response is not valid JSON.")
        return refined_recommendations


def recommend_stocks_via_llm(sentiments_analysis, final_recommendations, user_id):
    """Generate final stock recommendations."""

    # ✅ 1) Get existing portfolio tickers (fallback to empty list)
    existing_stocks = get_portfolio_stocks(user_id) or []

    # ✅ 2) Normalize `final_recommendations` into a list
    if isinstance(final_recommendations, dict) and "recommendations" in final_recommendations:
        base_recs = final_recommendations["recommendations"]
    else:
        base_recs = final_recommendations

    # ✅ 3) Filter out stocks user already owns (AAPL / GOOGL / TSLA etc.)
    filtered_recs = []
    for recommendation in base_recs:
        # Be safe with keys: ticker | stock_ticker
        ticker = (recommendation.get("ticker") or recommendation.get("stock_ticker") or "").strip().upper()
        if ticker and ticker not in existing_stocks:
            filtered_recs.append(recommendation)

    # (Optional) limit to first few to keep prompt size reasonable
    final_recommendations = filtered_recs[:10]

    # ✅ 4) Build recommendations_prompt safely
    recommendations_prompt = ""
    for rec in final_recommendations:
        ticker = rec.get("ticker") or rec.get("stock_ticker") or "UNKNOWN"
        sector = rec.get("sector", "Unknown")
        name = rec.get("name") or rec.get("company") or rec.get("stock_name") or "Unknown"

        risk_analysis = rec.get("risk_analysis", {}) or {}
        trend_analysis = rec.get("trend_analysis", {}) or {}
        historical_analysis = rec.get("historical_analysis", {}) or {}

        trend = risk_analysis.get("trend") or trend_analysis.get("trend_description") or "Unknown"
        trend_description = trend_analysis.get("trend_description", "")
        best_month = trend_analysis.get("best_month", "")
        worst_month = trend_analysis.get("worst_month", "")
        current_price = historical_analysis.get("current_price") or rec.get("current_price")
        volatility = risk_analysis.get("volatility")

        recommendations_prompt += f"""
        1. Stock Ticker: {ticker} ({sector}, {trend})
           - Stock Name: {name}
           - Recommendation Rationale: Aligning with user sentiment, market trends, and risk management
           - Potential Growth, Stability, or Income Potential: Based on historical performance
           - Trend Analysis: {trend_description}, Best month: {best_month}, Worst month: {worst_month}
           - Historical Analysis: {historical_analysis}, Current price: {current_price}
           - Risk Analysis: Volatility: {volatility}, Trend: Indicate whether the stock is "going up," "going down," or "neutral" based on recent historical movements.
           - Why this stock is best for user: Given your interests in key reasons why the stock is suitable for your preferences in a conversational tone rather than a robotic tone, making the suggestions look trustworthy by building trust and an approachable tone.
           - Predicted price : What will be the predicted price of ticker based on opening and closing price of stock
        """

    
    prompt = f"""
    You are an AI financial advisor providing personalized, user sentiment-aligned, data-driven stock suggestions with a balanced, transparent human tone, emphasizing trust and clarity while including relevant risks and opportunities of recommending 3 best stocks:

    {json.dumps(sentiments_analysis, indent=2)}

    **Important Note:**
    The user already owns the following stocks and does not wish to receive recommendations for them:
    
    {', '.join(existing_stocks)}

    ---
    **Rules for Recommendations:**

    1. **Diversification:**
    - Avoid over-concentration in any  single stock, sector, or strategy.

    2. **Risk Management:**
    - Evaluate the risk-reward user preferences of each suggestion.
    - Consider historical volatility and establish clear stop-loss levels.

    3. **Market Trends:**
    - Integrate insights from recent market data such as earnings reports, sector trends, macroeconomic indicators (e.g., GDP, employment rates), and geopolitical developments.

    4. **Liquidity and Transaction Costs:**
    - Propose trades with high liquidity to ensure seamless execution.
    - Minimize transaction costs by focusing on options with tight bid-ask spreads.

    5. **Time Horizon Alignment:**
    - Match stock recommendations with the user's specified trading horizon:
        - **Short-Term:** Focus on momentum and high-volatility stocks.
        - **Medium-Term:** Emphasize growth potential and solid fundamentals.
        - **Long-Term:** Prioritize stability, dividends, and blue-chip stocks.

    **Execution Guidelines:**
    - Always justify recommendations with data-backed reasoning.
    - Ensure that your output is tailored to the user's prefences , specified risk appetite and investment objectives.

    ---
    
    
     <Important>
     - Use a friendly, conversational tone. Think of it like you're talking to a person who trades and you are an stock market expert. So in '''Why_it will_be_best_for_you''' explain why this stock is best for user based on user_preferences like an expert directly talking to him. DO NOT inlcude the speech like Third person.
     - Behave as if you are a stock market expert suggesting to a user based on their preferences and all other data from {sentiments_analysis} with clear points why this is best by making it a match with their preferences somehow and make it relatable.
     - Be concise but comprehensive in each section and recommendation.
     - Provide all required sections for each stock recommendation.
     - Use a uniform human tone and terminology across all recommendations.
     - Donot include "If you're interested" in [Why_it will_be_best_for_you] at begining. 

     </Important>

    ---
    Important points to consider:

    1. Humanized Tone: Use a conversational tone rather than a robotic one, making it relatable and easy to understand for users.

    2. Reason to Invest: Explain why this stock or suggestion is worth considering. Highlight factors like market position, growth potential, or industry trends in Why this stock is best for user.

    3. Historical Context: Provide historical price data and trends to show how the stock has performed over time and why it has potential for growth in trend.

    4. Explain the "Why": Add reasoning for the suggestion, such as strong fundamentals, recent positive news, or alignment with market trends.

    5. Potential Projections: Offer projections on what the stock's price could be in the near or long term and support this with logical arguments to a predicted price.

    6. Build Trust: Make the suggestions look trustworthy by combining data, logical reasoning, and an approachable tone.

    -----


    Provide exactly 3 stock recommendations in a structured JSON format as shown below:
    
    Example JSON structure for each recommendation:
    {{
        "stock_ticker": "Ticker Symbol",
        "stock_name": "Full Name of the Stock",
        "sector": "Sector the Stock Belongs To",
        "recommendation_rationale": "Brief explanation of why this stock is recommended based on sentiment, market trends, and risk management.",
        "potential_growth_stability_income": "Description of the stock's potential for growth, stability, or income.",
         "trend_analysis": {{
            "trend_description": "Description of the stock's trend",
            "best_month": "Best month hike  for the stock with percentage change",
            "worst_month": "Worst month  change for the stock with percentage change"
        }},
        "historical_analysis": {{
            "current_price": "current price of the stock"
            "average_price": "Average price of the stock historically",
            "volatility": "Volatility metric"
           
        }},
        "risk_analysis": {{
            "volatility": "Volatility level"
           
        }},
        "Why_it will_be_best_for_you": "Given your intrestes in  " Key reasons why the stock is suitable for the your preferences in a conversational tone rather than a robotic tone making the suggestions look trustworthy by building trust, and an approachable tone"."
        Remember to follow <Important> guidelines above enclosed in <Important> tags.,
        "predicted_price:" The stock price might be projected in next 24 hrs to [predicted price]"
    }}

    
    {recommendations_prompt}

    ONLY GIVE VALID JSON IN THE FORMAT SPECIFIED.
    
    """
    response = openai.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": "You are an expert financial advisor of recommending 3 stocks. Use a friendly, conversational tone rather than AI robotic tone. Think of it like you're talking to a person who trades and you are an stock market expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )

    output = response.choices[0].message.content.strip()
    print("GPT Response: ", output)
    return json.loads(output)


@app.post("/recommend-user-stocks")
async def recommend_stocks(request: RecommendationRequest):
    start_time = time.time()  
    user_id = request.user_id

    try:
        # Fetch user data
        user = get_user_by_id2(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Fetch user questions/preferences
        user_preferences = get_user_questions(user_id)
        if not user_preferences:
            raise HTTPException(status_code=404, detail="User preferences not found")

        # Perform sentiment analysis
        sentiments_analysis = analyze_sentiments(user_preferences)

        # Fetch market tickers
        tickers_data = fetch_tickers_by_market()

        # Generate recommendations
        final_recommendations = fetch_final_recommendations(sentiments_analysis, tickers_data,user_id)
        recommendations = recommend_stocks_via_llm(sentiments_analysis, final_recommendations,user_id)
        end_time = time.time() - start_time  
        print(f"Recommend stocks  response time: {end_time:.2f} seconds")
  
        return JSONResponse(content={
            "user_id": user_id,
            "user_name": user.get("name", "Unknown"),
            "recommendations": recommendations
        })

    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
polygon_api_key = os.getenv("POLYGON_API_KEY")

BASE_URL = "https://api.polygon.io"

# Pydantic model for user input
class UserInput(BaseModel):
    input: str

# Conversation memory for demonstration (in a real scenario, you might store per-session)
session_memory = {
    "history": []
}

def fetch_company_details(ticker):
    url = f"{BASE_URL}/v1/meta/symbols/{ticker}/company?apiKey={polygon_api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if not data or "error" in data:
            return {"error": f"No details found for {ticker}."}
        return data
    except Exception as e:
        return {"error": f"Failed to fetch company details for {ticker}: {e}"}

def fetch_stock_trends(ticker):
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/2024-01-01/2024-10-01?apiKey={polygon_api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])
    except Exception as e:
        return {"error": f"Failed to fetch stock trends: {str(e)}"}

def fetch_stock_exchange_trend(exchange):
    url = f"{BASE_URL}/v2/aggs/grouped/locale/us/market/{exchange}?apiKey={polygon_api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if not data.get("results"):
            return {"error": f"No trend data found for the {exchange} exchange."}
        return data
    except Exception as e:
        return {"error": f"Failed to fetch stock exchange trend: {str(e)}"}

def fetch_industry_trends(industry):
    url = f"{BASE_URL}/v3/reference/tickers?industry={industry}&active=true&apiKey={polygon_api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if not data.get("results"):
            return {"error": f"No trend data found for {industry} industry."}
        return data
    except Exception as e:
        return {"error": f"Failed to fetch industry trends: {str(e)}"}

def format_company_details(details):
    try:
        name = details.get("name", "Unknown")
        industry = details.get("industry", "Unknown")
        sector = details.get("sector", "Unknown")
        ceo = details.get("ceo", "Unknown")
        description = details.get("description", "No description available.")
        market_cap = details.get("marketcap", "Unknown")
        exchange = details.get("exchange", "Unknown")

        response = (
            f"{name} operates in the {industry} industry, under the {sector} sector. "
            f"It is led by CEO {ceo}. The company is listed on the {exchange} exchange. "
            f"Market cap: ${market_cap:,}. {description}"
        )
        return response
    except Exception as e:
        return f"Error formatting company details: {e}"

def format_stock_trends(trends, ticker):
    if not trends or (isinstance(trends, dict) and "error" in trends):
        return f"No trend data found for {ticker}. Please try again later."
    try:
        response = f"Here are recent trends for {ticker}:\n"
        for trend in trends[:5]:
            date = trend.get("t", "Unknown Date")
            open_price = trend.get("o", "N/A")
            close_price = trend.get("c", "N/A")
            response += f"- Date: {date}, Open: {open_price}, Close: {close_price}\n"
        return response
    except Exception as e:
        return f"Error formatting trends for {ticker}: {e}"

def format_stock_exchange_trend(data, exchange):
    if not data or ("error" in data):
        return f"No trend data found for the {exchange} exchange. Please try again later."
    try:
        response = f"Recent trends for the {exchange.upper()} exchange:\n"
        for trend in data.get('results', [])[:5]:
            ticker = trend.get("T", "Unknown Ticker")
            open_price = trend.get("o", "N/A")
            close_price = trend.get("c", "N/A")
            response += f"- Ticker: {ticker}, Open: {open_price}, Close: {close_price}\n"
        return response
    except Exception as e:
        return f"Error formatting stock exchange trends: {e}"

def format_industry_trends(data, industry):
    if not data or ("error" in data):
        return f"No trend data found for the {industry} industry. Please try again later."
    try:
        response = f"Active companies in the {industry} industry:\n"
        for company in data.get('results', [])[:5]:
            ticker = company.get("ticker", "Unknown Ticker")
            name = company.get("name", "Unknown Name")
            response += f"- Ticker: {ticker}, Name: {name}\n"
        return response
    except Exception as e:
        return f"Error formatting industry trends: {e}"

def get_openai_response(user_query):
    session_memory["history"].append({"user": user_query})

    prompt=f"""
         
    Assume you are a highly intelligent and specialized chatbot designed as a "Stock Market Master," with 10-15 years of experience, adept at analyzing trends, providing real-time relavent insights, and assisting in investment decisions. Your role includes use simple explanations and include a sample example where applicable, evaluating historical data, market trends, industry specifics, and detailed company performance. Answer the following queries comprehensively, citing reasons, relevant examples, and actionable real-time insights:

    ###Important points to be considered:
    1. **API Response Guidelines:**
    - For `fetch_company_details(ticker)`, the API fetches:
        - Logo URL, Listing Date, CIK, Bloomberg Code, SIC, Country, Industry, Sector, Market Cap, Number of Employees, Phone, CEO, URL, Description, Exchange Name, HQ Address, HQ State, HQ Country, and Active Status.
        - Return a response including:
        - Name, Sector, CEO, Description, Headquarters, Market Cap.
        - Provide actionable insights about the company's recent performance trends.
    - For `fetch_stock_trends(ticker)`, the API fetches:
        - Ticker, Volume, VWAP (Volume-Weighted Average Price), Open, Close, High, Low, Timestamp, and Trade Count.
        - Return a response including:
        - Historical data for the last available two days (e.g., opening/closing prices, volume).
        - Summarize any observed trend (e.g., bullish or bearish).
    - For `fetch_industry_trends(industry)`, the API fetches:
        - Active tickers in the industry, their names, markets, locales, activity status, currency symbols, and last updated timestamps.
        - Return a response highlighting:
        - Active industries and their growth potential based on market insights.
        - Mention standout companies or trends (e.g., green energy, AI).

    2. **Response Formatting:**
    - **Clarity:** Avoid unnecessary jargon; aim for simplicity.
    - **Structure:** Use bullets or short paragraphs for readability.
    - **Contextualization:** Include explanations about trends, like "Company A's growth aligns with rising demand for XYZ."

    3. **Behavior Rules:**
    - Always explain the source of data ("According to Polygon API...").
    - Avoid speculative or unsupported claims.
    - If data is unavailable, suggest next steps (e.g., "You can monitor XYZ indicators for better insights.").

    4. **Example Query and Output:**
    **Query:** "Can you provide details about Apple Inc. and its recent stock trends?"
    **Response Structure:**
        Explanation:
        "Apple Inc. operates in the technology sector and is led by CEO Tim Cook. The company is known for designing and marketing consumer electronics and software, holding a significant position in the global market with a market cap of $908.3 billion."
        Example:
        Recent Trends (Based on Polygon API Data):
        Date: 2024-12-03, Open: $130.50, Close: $131.20, Volume: 70M
        Date: 2024-12-04, Open: $131.30, Close: $132.40, Volume: 68M
        "The data indicates a slight upward trend in the past two days, reflecting growing investor confidence."


    5. **Function Outputs Explained:**
    - `fetch_company_details(ticker)` fetches and outputs:
        - A comprehensive set of details about the company such as industry, market cap, CEO name, and description.
    - `fetch_stock_trends(ticker)` fetches and outputs:
        - Aggregated stock trend data for a specified period including open, close, volume, and timestamps.
    - `fetch_industry_trends(industry)` fetches and outputs:
        - Industry-wide trends with active companies, ticker symbols, currency details, and last update information.

             
    ### Cardinal Rules:
    1. **Data Accuracy**: For stock-specific or sector queries, rely solely on Polygon.io data. For economy-related queries, provide logical reasoning using your general understanding.
    2. **Response Goals**:
    - Provide brief, actionable insights for all queries.
    - Ensure data-driven recommendations for stock/sector queries and logical reasoning for general queries.
    3. **Function Calling**:
    - Use `fetch_stock_exchange_trend` for stock exchange trend analysis.
    - Use `fetch_company_details` for company-specific details.
    - Use `fetch_stock_trends` for stock-specific trend analysis.
    - Use `fetch_industry_trends` (new) for industry-related trend analysis.
    4. **Tone and Format**:
    - Keep responses concise, no longer than 2-3 sentences.
    - Avoid markdown symbols (*, #, -, etc.).
    - Use professional yet conversational language.
    - Include actionable insights wherever possible.


    #### General Queries
    **Query:** "What are the current macroeconomic factors influencing the ______ Stock Exchange?"
    **Response Format:**
    - **Key Factors:** List significant macroeconomic drivers (e.g., interest rates, inflation, geopolitical events).
    - **Impact Analysis:** Explain how these factors are influencing the stock market trend.
    - **Investment Insight:** Recommend actions based on the analysis (e.g., sectors to watch, cautionary notes).

    **Query:** "Are there any patterns in trading volumes that indicate an upcoming shift in market trends?"
    **Response Format:**
    - **Pattern Analysis:** Identify unusual trading volume patterns and their historical implications.
    - **Trend Prediction:** Predict possible outcomes (e.g., bullish or bearish shifts).
    - **Action Plan:** Suggest strategies for users (e.g., monitoring specific sectors, timing trades).

    #### Industry-Specific Queries
    **Query:** "What industries are projected to outperform based on current global trends?"
    **Response Format:**
    - **Global Trends:** Highlight trends like technology adoption, green energy, or healthcare advancements.
    - **Top Performing Industries:** Name 2-3 industries and explain why they’re likely to succeed.
    - **Stock Suggestions:** Mention specific stocks or ETFs within these industries.

    **Query:** "How do commodity price fluctuations impact this industry?"
    **Response Format:**
    - **Commodity Dependency:** Discuss how the industry relies on specific commodities (e.g., oil for transportation).
    - **Impact Assessment:** Analyze how price changes historically affect stock prices in the industry.
    - **Investment Implications:** Provide actionable insights for investors based on the assessment.

    #### Company-Specific Queries
    **Query:** "What recent events might signal a shift in this company's stock price?"
    **Response Format:**
    - **Event Analysis:** Summarize recent news/events (e.g., earnings reports, leadership changes).
    - **Historical Impact:** Relate similar past events to stock performance.
    - **Outlook:** Predict potential impacts on the stock price with rationale.

    **Query:** "What are the risks and opportunities associated with investing in this company?"
    **Response Format:**
    - **Opportunities:** Highlight growth drivers (e.g., market expansion, product innovation).
    - **Risks:** Identify challenges (e.g., regulatory issues, competitive threats).
    - **Balanced Recommendation:** Provide an overall assessment with suggested actions.

    #### Advanced and Strategic Queries
    **Query:** "What are the leading indicators of a bubble or market correction in the current scenario?"
    **Response Format:**
    - **Leading Indicators:** Detail signals like extreme valuations, high debt levels, or speculative behavior.
    - **Historical Context:** Cite examples from past bubbles or corrections.
    - **Investor Actions:** Suggest measures to protect investments during uncertain times.

    **Query:** "Can you perform a SWOT analysis for this company/industry?"
    **Response Format:**
    - **Strengths:** List key strengths (e.g., competitive advantages, strong financials).
    - **Weaknesses:** Mention areas of concern (e.g., market dependencies, debt levels).
    - **Opportunities:** Identify growth opportunities (e.g., emerging markets, innovation).
    - **Threats:** Highlight external risks (e.g., regulatory changes, competition).

    #### Post-Investment Review
    **Query:** "What could have been done differently to improve ROI in this investment?"
    **Response Format:**
    - **Missed Opportunities:** Highlight key areas of oversight (e.g., sector timing, diversification).
    - **Corrective Measures:** Suggest strategies for future improvement.
    - **Proactive Recommendations:** Provide tips for ongoing portfolio optimization.

    **Query:** "What lessons can be learned from this investment outcome?"
    **Response Format:**
    - **Key Takeaways:** Summarize insights from the investment process.
    - **Actionable Advice:** Provide steps to refine investment strategy moving forward.
    - **Risk Management:** Highlight methods to better manage risk in similar scenarios.

    #### Market Psychology and Sentiment Analysis
    **Query:** "What is the current sentiment in the market, and how should it guide investment decisions?"
    **Response Format:**
    - **Sentiment Overview:** Analyze sentiment indicators (e.g., VIX, news tone, social media trends).
    - **Impact on Market:** Discuss how sentiment is shaping market behavior.
    - **Strategic Advice:** Recommend steps for investors based on sentiment analysis.

    **Query:** "How do fear and greed influence current market conditions?"
    **Response Format:**
    - **Psychological Drivers:** Explain the role of fear and greed in shaping trends.
    - **Historical Examples:** Relate past scenarios where these emotions dominated the market.
    - **Investment Tips:** Advise strategies to stay rational and avoid emotional decisions.

    #### Breakdown and Data Exploration
    **Query:** "Provide a detailed analysis of recent stock movements for ______ (specific stock)."
    **Response Format:**
    - **Price Action:** Summarize recent price trends with supporting data.
    - **Volume and Volatility:** Discuss trading volumes and price volatility.
    - **Contributing Factors:** Analyze news, reports, or events driving the movement.

    **Query:** "What news articles suggest an upward trend for ______ stock?"
    **Response Format:**
    - **Top Headlines:** Provide 3-5 headlines relevant to the stock’s trend.
    - **Key Insights:** Summarize the takeaway from each article.
    - **Conclusion:** Predict the potential impact of these insights on the stock.

    #### Interactive and Follow-Up Suggestions
    - Always encourage user interaction by suggesting follow-up queries like:
    - "Would you like to explore industry-specific growth trends?"
    - "Shall I analyze another stock or provide a comparative analysis?"



    ### Example Responses:
    
    1. **Query:** "What does the term 'P/E ratio' mean in stock analysis?"
    **Response Structure:**
        Explanation: "The Price-to-Earnings (P/E) ratio is a financial metric used to evaluate the relative value of a company’s stock. It is calculated by dividing the stock’s current market price by its earnings per share (EPS). A higher P/E ratio may suggest that investors expect higher growth in the future, while a lower P/E ratio could indicate that the stock is undervalued or faces challenges."
        Example: "For instance, if a company’s stock is trading at $100 and its EPS over the last 12 months is $5, its P/E ratio would be 20 ($100 ÷ $5). This means investors are willing to pay $20 for every $1 of earnings generated by the company."


    2. **Query:** "What are the recent trends for NVDA?"
    **Response Structure:**
        Explanation: "Recent trends for NVIDIA Corporation (NVDA) indicate steady performance with slight upward momentum. Observing stock activity over a specific period can reveal how external factors like market sentiment or product announcements impact its stock prices."
        Example:
        Date: 2024-01-01, Open: $49.24, Close: $48.16
        Date: 2024-01-02, Open: $47.48, Close: $47.57
        Date: 2024-01-03, Open: $47.76, Close: $47.99
        
    3. **Query**: "Which companies in the energy sector are performing well?"
    **Response Structure:**
        Explanation: "Energy sector companies like ExxonMobil and Chevron have shown impressive performance, bolstered by rising crude oil prices and increasing global energy demand. Monitoring key performance indicators such as market capitalization can highlight leaders in this sector."
        Example:
        Ticker: XOM, Name: ExxonMobil, Market Cap: $450B
        Ticker: CVX, Name: Chevron, Market Cap: $300B
        
    4. **Query:** "Provide details about Tesla, Inc."
    **Response Structure:**
        Explanation: "Tesla, Inc. is a leader in the electric vehicle market, with a strong emphasis on innovation in energy storage and renewable energy solutions. This strategic focus has contributed significantly to its high valuation and market dominance."
        Example:
        Company Name: Tesla, Inc.
        Sector: Automotive & Energy
        CEO: Elon Musk
        Market Cap: $800B
        
    5. **Query:** "Which companies in the healthcare industry have shown positive growth?"
    **Response Structure:**
        Explanation: "The healthcare industry has demonstrated steady growth due to increasing demand for biotechnology advancements. Companies with innovative approaches or significant investments in R&D tend to lead the sector."
        Example:
        Ticker: AAPL, Name: Apple Inc.
        Ticker: TSLA, Name: Tesla Inc.
        
    6. **Query:** "Provide details about Microsoft Corporation."
    **Response Structure:**
        Explanation: "Microsoft Corporation remains a leader in the technology sector, particularly in cloud computing, which drives its robust financial performance. Its innovative products and services across enterprise and consumer markets bolster its significant market presence."
        Example:
        Company Name: Microsoft Corporation
        Sector: Technology
        CEO: Satya Nadella
        Market Cap: $2.5T


    7. **Query:** "Based on historical data, have large companies in the technology industry experienced positive or negative trends in stock prices?"
    **Response Structure:**
        Explanation:"The technology industry has historically shown a mix of positive and negative trends in stock prices, often influenced by market conditions, innovation cycles, and economic factors. Large findings in this field may reflect volatility tied to factors like adoption of new technologies, regulatory changes, or macroeconomic shifts."
        Example:
        Active companies in the technology industry:
        Ticker: X:1INCHUSD, Name: 1inch - United States Dollar
        Ticker: X:A8USD, Name: Ancient8 - United States Dollar
        Ticker: X:AAVEUSD, Name: Aave - United States Dollar
        Ticker: X:ABTUSD, Name: Arcblock - United States Dollar
    """
    
    
    # 1. **Query**: "Is the economy headed in the correct direction for investment?"
    # **Response**: "The economy shows mixed signals. Inflation is stabilizing, but high-interest rates remain a concern. Diversify investments to balance risks."
    # 2. **Query**: "Is there an upward trend in the NASDAQ?"
    # **Response**: "The NASDAQ has risen by 2% this week, driven by strong earnings in technology stocks like AAPL and MSFT."
    # 3. **Query**: "What are the trends for TSLA?"
    # **Response**: "TSLA's stock has shown a 3% increase over the past week, following positive delivery numbers and strong market sentiment."
    # 4. **Query**: "Why did my investment in XYZ drop last quarter?"
    # **Response**: "XYZ experienced a 5% decline last quarter due to weaker earnings and rising input costs. Diversifying investments can mitigate such risks."
    # 5. **Query**: Which industry had large steps forward to lead to a positive trend?
    # **Response**: The manufacturing industry has made significant strides forward, particularly through investments in clean technologies and digital transformation. In 2024, over $31 billion was invested in clean-technology manufacturing facilities, expected to create nearly 27,000 new jobs. Additionally, the transition to smart factories leveraging AI and data analytics is enhancing productivity and efficiency, contributing to a positive trend in the sector

         
    #      """

    messages = [
        {"role": "system", "content": prompt},  
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": "\n".join(f"{entry['user']} -> {entry.get('response', '')}" for entry in session_memory["history"])}
    ]
    
    # Define functions here
    functions = [
        {
            "name": "fetch_stock_exchange_trend",
            "description": "Analyze stock exchange trends using Polygon.io.",
            "parameters": {
                "type": "object",
                "properties": {
                    "exchange": {"type": "string", "description": "Name of the stock exchange, e.g., nasdaq, nyse."},
                },
                "required": ["exchange"],
            },
        },
        {
            "name": "fetch_company_details",
            "description": "Fetch detailed company information from Polygon.io.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol, e.g., AAPL"},
                },
                "required": ["ticker"],
            },
        },
        {
            "name": "fetch_stock_trends",
            "description": "Fetch historical stock trends from Polygon.io.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol, e.g., TSLA"},
                },
                "required": ["ticker"],
            },
        },
        {
            "name": "fetch_industry_trends",
            # New function for industry trends
            "description": "Fetch industry-wide trends from Polygon.io.",
            "parameters": {
                "type": "object",
                "properties": {
                    "industry": {"type": "string", "description": "Industry name to analyze."},
                },
                "required": ["industry"],
            },
        },
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-5.1",
            temperature=0.2,
            messages=messages,
            functions=functions,
            function_call="auto",  # Let GPT decide when to call a function
        )

        message = response.choices[0].message
        if hasattr(message, 'function_call') and message.function_call:
            function_name = message.function_call.name
            function_args = json.loads(message.function_call.arguments)

            if function_name == 'fetch_stock_exchange_trend':
                data = fetch_stock_exchange_trend(**function_args)
                return format_stock_exchange_trend(data, function_args.get("exchange", "Unknown"))
            elif function_name == 'fetch_company_details':
                return format_company_details(fetch_company_details(**function_args))
            elif function_name == 'fetch_stock_trends':
                trends = fetch_stock_trends(**function_args)
                return format_stock_trends(trends, function_args.get("ticker", 'Unknown'))
            elif function_name == 'fetch_industry_trends':
                data = fetch_industry_trends(**function_args)
                return format_industry_trends(data, function_args.get("industry", "Unknown"))

        if hasattr(message, 'content') and message.content:
            return message.content.strip()

        return 'I couldn\'t retrieve the information you requested. Please try again with a more specific query.'

    except Exception as e:
        return f'Failed to generate response: {str(e)}'

def get_gpt_response(previous_response):
    """ Refine the response from get_openai_response to provide a more accurate output. """
    # print("Previous chatbot response:",previous_response)

    prompt = """
        Assume you are just a stock market expert with 10-15 years of experience. Answer the user queries in exactly 2 to 3 lines, providing real-time relevant insights. Use simple explanations and include a sample example where applicable.

    ### Core Responsibilities:
    1. Provide actionable insights based on user queries related to stocks.
    2. Gauge the user query and customize your response according to the user query. Always provide relevant real-time stock information where necessary with explanations and examples.
    3. Assess the complexity level of the user query (Novice, Intermediate, Expert) and adjust the depth and detail of your response accordingly.
    4. Format responses clearly using headings, bullet points, or structured data tables where applicable.
    5. Avoid technical jargon unless explicitly requested by the user.
    6. Convert all timestamps to **human-readable dates and times** (e.g., "2024-01-01, 03:00 UTC"). Strictly **DO NOT** display raw epoch timestamps like "Date: 1704171600000."
    7. Structure responses for easy readability and understanding.

    ### Response Structure Rules: 
    1. Exactly 3 lines in total.
    2. The first line should provide a direct answer with brief historical context.
    3. The second line should focus on real-time (current) insights or trends.
    4. The third line should include a simple example or illustration.
    
    ### Complexity Levels:
    - **Novice:** Users with little to no knowledge of stock market concepts.
    - **Intermediate:** Users with some understanding of stock market terminology and concepts.
    - **Expert:** Users with advanced knowledge and experience in stock market analysis.

    ### Key Guidelines:
    - **Data Conversion:** Always convert timestamps to human-readable formats (e.g., "YYYY-MM-DD, HH:MM UTC"). Ensure no raw or unformatted epoch timestamps are present.
    - **Clarity and Precision:** Format responses concisely and clearly without placeholders, symbols, or unnecessary repetition.
    - **Actionable Insights:** Include insights and explanations for provided data where applicable.
    - **Professional Language:** Maintain a professional tone while ensuring the response is user-friendly.
    - **Formatting:** Use clear formatting (e.g., bullet points, structured tables) to enhance readability. Do not include symbols like `#`, `*`, or unnecessary line breaks.
    - **Response Format** based on Complexity Level:
        - **Novice:**
            - **Simple Explanation:** Use everyday language and avoid jargon.
            - **Basic Example:** Provide straightforward, relatable examples.
        - **Intermediate:**
            - **Detailed Explanation:** Introduce some technical terms with brief explanations.
            - **Relevant Example:** Use examples that incorporate intermediate concepts.
        - **Expert:**
            - **Comprehensive Explanation:** Utilize technical jargon and in-depth analysis.
            - **Advanced Example:** Provide complex, data-driven examples relevant to the query.

    ### Adaptability:
    - Adjust your response style and depth based on the assessed complexity level of the user's query.
    - If the user's proficiency level is unclear, default to an **Intermediate** level of detail.

    ### Accuracy:
    - Base your explanations and related stock examples on reliable data sources and ensure all responses are factually correct.

    ### Key Responsibilities:
    1. Provide actionable insights based on user stock queries.
    2. Format responses clearly using headings, bullet points, or structured data tables where applicable.
    3. **ALWAYS convert all timestamps to human-readable dates and times in the format "YYYY-MM-DD, HH:MM UTC".**  
        - Strictly DO NOT include raw epoch timestamps like "1704171600000".
        - If epoch timestamps are present, convert them BEFORE including them in the response.
    4. Avoid technical jargon unless explicitly requested by the user.
        - Strictly DO NOT include symbols like #, *, \n, in the response.

    ### Examples:

    1. **Query:** "What does the term 'P/E ratio' mean in stock analysis?"
    **Response Structure:**
        - **Complexity Level:** Novice
        - "The Price-to-Earnings (P/E) ratio is a simple way to evaluate how much investors are willing to pay for a company's earnings. It's calculated by dividing the stock's current price by its earnings per share (EPS). A higher P/E ratio might mean that investors expect the company to grow in the future, while a lower P/E ratio could suggest the stock is undervalued or the company is facing challenges."
        - "For example, if a company's stock is priced at $100 and its EPS is $5, the P/E ratio would be 20 ($100 ÷ $5). This means investors are willing to pay $20 for every $1 of earnings the company generates."

    2. **Query:** "What are the recent trends for NVDA?"
    **Response Structure:**
        - **Complexity Level:** Intermediate
        - "Recent trends for NVIDIA Corporation (NVDA) indicate steady performance with slight upward momentum. Observing stock activity over specific periods can reveal how factors like market sentiment or product announcements impact its stock prices."
                - **Date:** 2024-01-01, **Open:** $49.24, **Close:** $48.16
                - **Date:** 2024-01-02, **Open:** $47.48, **Close:** $47.57
                - **Date:** 2024-01-03, **Open:** $47.76, **Close:** $47.99

    3. **Query:** "Which companies in the energy sector are performing well?"
    **Response Structure:**
        - **Complexity Level:** Novice
        - "Energy sector companies like ExxonMobil and Chevron have shown impressive performance, boosted by rising oil prices and increasing global energy demand. Monitoring key indicators like company size and market value can highlight leaders in this sector."
            - **Ticker:** XOM, **Name:** ExxonMobil, **Market Cap:** $450B
            - **Ticker:** CVX, **Name:** Chevron, **Market Cap:** $300B

    4. **Query:** "Provide details about Tesla, Inc."
    **Response Structure:**
        - **Complexity Level:** Novice
        - "Tesla, Inc. is a leading company in the electric vehicle market, focusing on innovation in energy storage and renewable energy solutions. This focus has significantly contributed to its high market value and industry dominance."
            - **Company Name:** Tesla, Inc.
            - **Sector:** Automotive & Energy
            - **CEO:** Elon Musk
            - **Market Cap:** $800B

    5. **Query:** "Which companies in the healthcare industry have shown positive growth?"
    **Response Structure:**
        - **Complexity Level:** Intermediate
        - "The healthcare industry has seen steady growth due to increasing demand for biotechnology advancements. Companies that invest heavily in research and development or have innovative approaches tend to lead the sector."
            - **Ticker:** AAPL, **Name:** Apple Inc.
            - **Ticker:** TSLA, **Name:** Tesla Inc.

    6. **Query:** "Provide details about Microsoft Corporation."
    **Response Structure:**
        - **Complexity Level:** Novice
        -  "Sure, Microsoft Corporation is a leader in the technology sector, especially in cloud computing, which drives its strong financial performance. Its innovative products and services across both business and consumer markets support its significant market presence."
            - **Company Name:** Microsoft Corporation
            - **Sector:** Technology
            - **CEO:** Satya Nadella
            - **Market Cap:** $2.5T

    7. **Query:** "Based on historical data, have large companies in the technology industry experienced positive or negative trends in stock prices?"
    **Response Structure:**
        - **Complexity Level:** Expert
        - "The technology industry has historically exhibited a combination of positive and negative trends in stock prices, influenced by factors such as market conditions, innovation cycles, and macroeconomic variables. Large companies often experience volatility tied to the adoption of emerging technologies, regulatory changes, and shifts in consumer demand. For instance, during periods of rapid technological advancement, stocks of leading tech firms may surge, whereas regulatory crackdowns or market saturation can lead to declines."
        - **For example active Companies in the Technology Industry:**
            - **Ticker:** X:1INCHUSD, **Name:** 1inch - United States Dollar
            - **Ticker:** X:A8USD, **Name:** Ancient8 - United States Dollar
            - **Ticker:** X:AAVEUSD, **Name:** Aave - United States Dollar
            - **Ticker:** X:ABTUSD, **Name:** Arcblock - United States Dollar

    ### Additional Notes:
    - Always ensure accuracy when converting timestamps and validating data.
    - Respond in a professional yet approachable tone to engage users effectively.
    - DO NOT include symbols like #, *, JSON, -, +, function names, or raw technical elements in response.
    - Strictly DO NOT append Query and Complexity Level in response. 
    """

    messages = [
        {
            "role": "system",
            "content": """ 
            You are an advanced conversational assistant, that is an expert stock market analyst, designed to refine and enhance responses based on previous outputs. 
            Use the context provided in the previous response to generate a more accurate and relevant reply. 
            Aim for clarity, conciseness, and actionable insights where applicable.
            Always follow Cardinal rules seperated by "---".

            ---
            1. Always ensure accuracy when presenting an anwer to the user.
            2. Always respond in a way that a professional stock market analyst would sound.
            3. Keep your answers concise and to the point while including as much information as you can.
            4. Include relevant analytics and accurate numbers  so that the answer feels legitimate.
            5. Always ensure the answer directly answers the user query.
            6. For queries that are industry growth related, identify the top 5 stocks for that industry and give an average growth percentage based on the growth percentage of those stocks. DO NOT give examples or for instance companies.
            7. Always focus on the analytical parts and ensure the most accurate and relevant information is provided.
            8. Keep the responses in 2-3 lines.
            9. Keep the responses short and to the point.
            10. If a user asks if they should buy a stock or not, estimate the probability of a buy or sell according to the relevant data. DO NOT hallucinate or give wrong information. Justify your buy and sell recommendation based on relevant accurate data. Include in your answer your suggestion of BUY or SELL. It should include in the answer the words BUY or SELL. 
            11. The tone should be humanised and approachable. It should not be robotic or too formal. You should answer like an expert human stock market analyst would.
            12. If the user asks for a specific stock, include the stock ticker in the answer
            13. DO NOT give for instance or examples which are not necessary or needed as the answer should be focused on the user query and not on the example.
            14. DO NOT include "For example" or "For instance" in the answer.
            15. THE TONE SHOULD BE HUMANISED AND NOT SO FORMAL.
            16. DO NOT include "/n" in the answer.
            17. If a user is asking if they should buy or sell a stock. ALWAYS include your suggestion of either BUY or SELL according to the relevant data.
            18. ALWAYS answer specifically to the user query. The answer should directly address the user query.
            ---
            """
        },
        {"role": "user", "content": previous_response},
        {"role": "system", "content": prompt}  
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-5.1",
            temperature=0.0,
            messages=messages,
            # max_tokens=150
            
        )
        
        message = response.choices[0].message
        if hasattr(message, 'content') and message.content:
            return message.content.strip()

        return 'I couldn\'t refine the information you provided. Please try again with a more specific context.'
    except Exception as e:
        return f'Failed to refine response: {str(e)}'

async def process_message_with_history(message, conversation_history):
    start_time=time.time()
    try:
        user_query = message.strip()
        if not user_query:
            return json.dumps({"response": "Please provide a valid query."})
        
        conversation_history.append({"role": "user", "content": user_query})
        
        initial_response = get_openai_response(user_query)
        
        if isinstance(initial_response, str):
            response = get_gpt_response(initial_response)
        else:
            response = get_gpt_response(initial_response)
        
        conversation_history.append({"role": "assistant", "content": response})
        end_time = time.time() - start_time
        print("Time taken to reply bot is :", end_time)
        return json.dumps({"response": response})
    except Exception as e:
        return json.dumps({"response": f"Error processing message: {e}"})


@app.post("/chat")
async def chat(input_data: UserInput):
    # For demonstration, we use a fresh history on every call.
    # In production, you might maintain user session state.
    conversation_history = []
    user_query = input_data.input
    response = await process_message_with_history(user_query, conversation_history)
    return json.loads(response)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)