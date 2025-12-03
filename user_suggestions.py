from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv
import os 
import openai
import requests
import json
import uvicorn 
from datetime import datetime
from bson import ObjectId
from bson.errors import InvalidId

load_dotenv()

app = FastAPI()


class RecommendationRequest(BaseModel):
    user_id: str

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")   
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
MONGO_DB_HOST = os.getenv("MONGO_DB_HOST")
MONGO_DB_PORT = int(os.getenv("MONGO_DB_PORT", 27017))
MONGO_DB_USER = os.getenv("MONGO_DB_USER")
MONGO_DB_PASSWORD = os.getenv("MONGO_DB_PASSWORD")

   
openai.api_key = OPENAI_API_KEY

mongo_client = pymongo.MongoClient(
    host=MONGO_DB_HOST,
    port=MONGO_DB_PORT,
    username=MONGO_DB_USER,
    password=MONGO_DB_PASSWORD
)

db = mongo_client['ima-user']
collection = db['users']


def get_user_by_id(user_id: str):
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
    user = get_user_by_id(user_id)  
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
        model="gpt-4o",
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


def analyze_risk(ticker):
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

def fetch_final_recommendations(sentiments_analysis, tickers_data):
    """Filter and recommend stocks based on user sentiment and risk analysis with an added LLM layer."""
    recommendations = []

    for ticker_data in tickers_data[:20]: 
        ticker = ticker_data["ticker"]
        name = ticker_data.get("name", "Unknown")
        sector = ticker_data.get("market", "Unknown")

        # Fetch historical and risk analysis
        historical_data = fetch_historical_data(ticker, "2024-01-01", "2024-12-31")
        trend_analysis = analyze_trend(historical_data)
        historical_analysis = analyze_historical_data(historical_data)
        risk_data = analyze_risk(ticker)
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
        model="gpt-4o",
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


def recommend_stocks_via_llm(sentiments_analysis, final_recommendations):
    """Generate final stock recommendations."""
    recommendations_prompt = ""

    # Build recommendations text dynamically
    for rec in final_recommendations:
        recommendations_prompt += f"""
        1. Stock Ticker: {rec['ticker']} ({rec['sector']}, {rec['risk_analysis']['trend']})
           - Stock Name: {rec['name']}
           - Recommendation Rationale: Aligning with user sentiment, market trends, and risk management
           - Potential Growth, Stability, or Income Potential: Based on historical performance
           -Trend Analysis: {rec['trend_analysis']['trend_description'],rec['trend_analysis']['best_month'],rec['trend_analysis']['worst_month']}
           - Historical Analysis: {rec['historical_analysis'],rec['current_price']}
           - Risk Analysis: Volatility: {rec['risk_analysis']['volatility']}, Trend: Indicate whether the stock is "going up," "going down," or "neutral" based on recent historical movements.
           - Why this stock is best for user: Given your intrestes in Key reasons why the stock is suitable for the your preferences in a conversational tone rather than a robotic tone making the suggestions look trustworthy by building trust, and an approachable tone .
           - Predicted price : What will be the predicted price of ticker based on opening and closing price of stock
        """

    
    prompt = f"""
    You are an AI financial advisor providing personalized, user sentiment-aligned, data-driven stock suggestions with a balanced, transparent human tone, emphasizing trust and clarity while including relevant risks and opportunities of recommending 3 best stocks:

    {json.dumps(sentiments_analysis, indent=2)}

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
    
    """

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert financial advisor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        response_format={"type": "json_object"}
    )
    output = response.choices[0].message.content.strip()
    return json.loads(output)


@app.post("/recommend-user-stocks")
async def recommend_stocks(request: RecommendationRequest):

    user_id = request.user_id

    try:
        # Fetch user data
        user = get_user_by_id(user_id)
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
        final_recommendations = fetch_final_recommendations(sentiments_analysis, tickers_data)
        recommendations = recommend_stocks_via_llm(sentiments_analysis, final_recommendations)

        return JSONResponse(content={
            "user_id": user_id,
            "user_name": user.get("name", "Unknown"),
            "recommendations": recommendations
        })

    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)