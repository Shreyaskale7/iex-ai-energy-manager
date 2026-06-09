# API Integration Examples

This document outlines how to interact with the FastAPI backend across different platforms.

## 1. cURL Examples

**Check Health (No API Key required)**
```bash
curl -X GET "http://localhost:8000/health"
```

**Get Market Status**
```bash
curl -X GET "http://localhost:8000/market-status?limit=10" \
     -H "X-API-Key: dev-api-key"
```

**Get 24-Hour Forecast**
```bash
curl -X GET "http://localhost:8000/forecast/day" \
     -H "X-API-Key: dev-api-key"
```

**Calculate Smart Load Schedule**
```bash
curl -X POST "http://localhost:8000/decision/schedule" \
     -H "X-API-Key: dev-api-key" \
     -H "Content-Type: application/json" \
     -d '{
           "forecast_type": "24-Hour",
           "devices": [
             {"device_id": "A1", "name": "HVAC", "category": "Flexible", "power_kw": 50.0}
           ]
         }'
```

## 2. Postman Configuration

1. Create a new collection for `IEX Smart Energy`.
2. Set a Collection Variable: `baseUrl` = `http://localhost:8000`.
3. Set the Authorization tab to `API Key`:
   - **Key**: `X-API-Key`
   - **Value**: `dev-api-key`
   - **Add to**: `Header`
4. Use standard routes (e.g., `{{baseUrl}}/forecast/latest`).

## 3. Frontend Fetch Example (JavaScript/TypeScript)

```javascript
const API_KEY = process.env.NEXT_PUBLIC_IEX_API_KEY;
const BASE_URL = 'http://localhost:8000';

// Fetch specific date forecast
async function fetchForecastDate(dateString) {
  const response = await fetch(`${BASE_URL}/forecast/date/${dateString}`, {
    method: 'GET',
    headers: {
      'X-API-Key': API_KEY,
      'Content-Type': 'application/json'
    }
  });

  if (!response.ok) {
    throw new Error('Failed to fetch forecast');
  }

  const data = await response.json();
  return data.points;
}

// Generate Schedule
async function generateSchedule(devices) {
  const response = await fetch(`${BASE_URL}/decision/schedule`, {
    method: 'POST',
    headers: {
      'X-API-Key': API_KEY,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      forecast_type: "24-Hour",
      devices: devices
    })
  });
  
  return await response.json();
}
```
