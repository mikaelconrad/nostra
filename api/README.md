# Cryptocurrency Investment API Documentation

## Overview
REST API for the Cryptocurrency Investment Recommendation System.

## Base URL
```
http://localhost:5000/api
```

## Endpoints

### Health Check
```
GET /api/health
```
Returns the API status and version.

### Cryptocurrency Data

#### Get Price Data
```
GET /api/crypto/{symbol}/data?days=30
```
- `symbol`: BTC, ETH, or XRP
- `days`: Number of days of data to return (optional, default: 30)

#### Update Price Data
```
POST /api/crypto/{symbol}/update
```
Body:
```json
{
  "days": 7
}
```

### Portfolio Management

#### Get Portfolio
```
GET /api/portfolio
```
Returns current portfolio holdings, metrics, and performance.

#### Buy Cryptocurrency
```
POST /api/portfolio/buy
```
Body:
```json
{
  "symbol": "BTC",
  "amount": 0.001,
  "price": 50000
}
```

#### Sell Cryptocurrency
```
POST /api/portfolio/sell
```
Body:
```json
{
  "symbol": "BTC",
  "amount": 0.001,
  "price": 50000
}
```

#### Get Transactions
```
GET /api/portfolio/transactions
```
Returns transaction history and summary.

### Analytics

#### Get Recommendations
```
GET /api/recommendations
```
Returns investment recommendations for all cryptocurrencies.

#### Get Sentiment Analysis
```
GET /api/sentiment/{symbol}?days=7
```
- `symbol`: BTC, ETH, or XRP
- `days`: Number of days of sentiment data (optional, default: 7)

### Reports

#### Generate Daily Report
```
POST /api/reports/generate
```
Generates and returns the path to a daily portfolio report.

### Data Processing

#### Process All Data
```
POST /api/process
```
Runs data processing pipeline for all cryptocurrencies.

## Error Responses

All errors return JSON with an `error` field:
```json
{
  "error": "Error message"
}
```

HTTP Status Codes:
- 200: Success
- 400: Bad Request
- 422: Validation Error
- 500: Internal Server Error

## Example Usage

```python
import requests

# Get portfolio data
response = requests.get('http://localhost:5000/api/portfolio')
data = response.json()

# Buy Bitcoin
response = requests.post('http://localhost:5000/api/portfolio/buy', json={
    'symbol': 'BTC',
    'amount': 0.001,
    'price': 50000
})
result = response.json()
```