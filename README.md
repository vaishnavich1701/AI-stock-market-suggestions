# AI-stock-market-suggestions

This is a FastAPI-based project for managing user portfolios and generating stock recommendations using OpenAI's GPT API. It integrates with MongoDB and the Polygon.io API to fetch stock data and perform risk analysis.

## Features

- **Portfolio Management**: Fetch user and portfolio data from MongoDB.
- **Stock Analysis**: Evaluate stock trends, risk levels, and historical performance.
- **Recommendations**: Generate personalized stock suggestions and predictions using OpenAI GPT.
- **API Integration**: Connects with Polygon.io for real-time and historical stock data.

## Setup Instructions

### Prerequisites

- **Python**: Version 3.8 or higher
- **MongoDB**: A running MongoDB server
- **API Keys**:
  - OpenAI
  - Polygon.io

### Installation

1. **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Create a `.env` file with the following variables:**

    ```makefile
    OPENAI_API_KEY=<your_openai_api_key>
    POLYGON_API_KEY=<your_polygon_api_key>
    MONGO_DB_HOST=<your_mongo_host>
    MONGO_DB_PORT=<your_mongo_port>
    MONGO_DB_USER=<your_mongo_user>
    MONGO_DB_PASSWORD=<your_mongo_password>
    ```

4. **Run the application:**

    ```lua
    uvicorn api:app --reload
    ```

    The API will be available at [http://localhost:8000](http://localhost:8000).

## API Endpoints

### Fetch Portfolio Suggestions

- **Endpoint**: `/portfolio/suggestions/`
- **Method**: `GET`
- **Query Parameters**:
  - `user_id` (string): The ObjectId of the user.
- **Response**: Returns stock recommendations and predictions according to the user's portfolios.

