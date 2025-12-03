import os
import json
import time
from datetime import datetime
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import requests
import openai
from dotenv import load_dotenv
import uvicorn


# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
polygon_api_key = os.getenv("POLYGON_API_KEY")

BASE_URL = "https://api.polygon.io"
app = FastAPI()

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
            model="gpt-4o-mini",
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
            model="gpt-4o-mini",
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
    uvicorn.run(app, host="localhost", port=8765)
