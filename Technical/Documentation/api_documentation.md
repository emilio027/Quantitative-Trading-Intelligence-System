# Quantitative Trading Intelligence System
## API Documentation

### Version 2.0.0 Enterprise
### Author: API Documentation Team
### Date: August 2025

---

## Overview

The Quantitative Trading Intelligence System provides comprehensive RESTful APIs for algorithmic trading, market analysis, portfolio optimization, and risk management. All APIs are designed for institutional-grade performance with sub-millisecond latency and 99.99% availability.

**Base URL**: `https://api.quantrading.enterprise.com/v2`
**Authentication**: Bearer Token (OAuth 2.0) + API Key
**Rate Limiting**: 10,000 requests/minute per API key
**WebSocket**: `wss://ws.quantrading.enterprise.com/v2`

## Authentication

### OAuth 2.0 + API Key Authentication

All API requests require both a bearer token and API key:

```bash
Authorization: Bearer {access_token}
X-API-Key: {api_key}
```

### Get Access Token

**Endpoint**: `POST /auth/token`

**Request**:
```json
{
  "grant_type": "client_credentials",
  "client_id": "your_client_id",
  "client_secret": "your_client_secret",
  "scope": "trading:read trading:write portfolio:manage market-data:read"
}
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "scope": "trading:read trading:write portfolio:manage market-data:read"
}
```

## Market Data APIs

### 1. Real-Time Market Data

#### Get Current Market Data

**Endpoint**: `GET /market-data/quotes`

**Description**: Retrieve real-time quotes for specified symbols with sub-millisecond latency.

**Query Parameters**:
- `symbols` (string): Comma-separated list of symbols (e.g., "AAPL,GOOGL,MSFT")
- `fields` (string): Requested fields (bid,ask,last,volume,timestamp)
- `format` (string): Response format (json, msgpack, protobuf)

**Example Request**:
```bash
GET /market-data/quotes?symbols=AAPL,GOOGL,MSFT&fields=last,volume,timestamp
```

**Response**:
```json
{
  "quotes": [
    {
      "symbol": "AAPL",
      "last": 182.45,
      "volume": 45678923,
      "timestamp": "2025-08-18T15:30:45.123456Z",
      "bid": 182.44,
      "ask": 182.46,
      "bid_size": 1200,
      "ask_size": 800
    },
    {
      "symbol": "GOOGL", 
      "last": 2847.91,
      "volume": 12345678,
      "timestamp": "2025-08-18T15:30:45.123789Z",
      "bid": 2847.89,
      "ask": 2847.93,
      "bid_size": 500,
      "ask_size": 300
    }
  ],
  "server_timestamp": "2025-08-18T15:30:45.124Z",
  "latency_ms": 0.8
}
```

#### Historical Market Data

**Endpoint**: `GET /market-data/history`

**Description**: Retrieve historical OHLCV data with technical indicators.

**Query Parameters**:
- `symbol` (string): Stock symbol
- `start_date` (string): Start date (YYYY-MM-DD)
- `end_date` (string): End date (YYYY-MM-DD)
- `interval` (string): Data interval (1m, 5m, 15m, 1h, 1d)
- `indicators` (string): Technical indicators to include

**Example Request**:
```bash
GET /market-data/history?symbol=AAPL&start_date=2025-07-01&end_date=2025-08-18&interval=1d&indicators=sma,rsi,macd
```

**Response**:
```json
{
  "symbol": "AAPL",
  "interval": "1d",
  "data": [
    {
      "timestamp": "2025-07-01T00:00:00Z",
      "open": 180.25,
      "high": 182.45,
      "low": 179.80,
      "close": 181.90,
      "volume": 56789012,
      "indicators": {
        "sma_20": 179.45,
        "rsi": 67.8,
        "macd": 1.23,
        "macd_signal": 1.18,
        "macd_histogram": 0.05
      }
    }
  ],
  "total_records": 35,
  "processing_time_ms": 12.4
}
```

### 2. WebSocket Market Data Feed

#### Subscribe to Real-Time Data

**WebSocket Endpoint**: `wss://ws.quantrading.enterprise.com/v2/market-data`

**Subscribe Message**:
```json
{
  "action": "subscribe",
  "channel": "quotes",
  "symbols": ["AAPL", "GOOGL", "MSFT"],
  "throttle_ms": 100
}
```

**Real-Time Update**:
```json
{
  "channel": "quotes",
  "symbol": "AAPL",
  "data": {
    "last": 182.46,
    "volume": 45678924,
    "timestamp": "2025-08-18T15:30:45.125456Z"
  }
}
```

## Trading APIs

### 1. Strategy Management

#### Create Trading Strategy

**Endpoint**: `POST /strategies`

**Description**: Create and configure a new trading strategy with ML models.

**Request Body**:
```json
{
  "name": "LSTM Momentum Strategy",
  "description": "Deep learning momentum strategy using LSTM networks",
  "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
  "strategy_type": "MOMENTUM",
  "model_config": {
    "primary_model": "lstm_ensemble_v2.1",
    "features": [
      "price_momentum_5d",
      "volume_momentum_10d", 
      "volatility_ratio",
      "rsi_divergence",
      "macd_signal"
    ],
    "lookback_period": 60,
    "prediction_horizon": 5,
    "confidence_threshold": 0.75
  },
  "risk_parameters": {
    "max_position_size": 0.05,
    "stop_loss": 0.02,
    "take_profit": 0.06,
    "max_drawdown": 0.15,
    "position_timeout_hours": 72
  },
  "execution_parameters": {
    "order_type": "LIMIT",
    "execution_style": "TWAP",
    "max_participation_rate": 0.15,
    "urgency": "MEDIUM"
  }
}
```

**Response** (201 Created):
```json
{
  "strategy_id": "STRAT-LSTM-001",
  "name": "LSTM Momentum Strategy",
  "status": "CREATED",
  "model_validation": {
    "backtest_results": {
      "annual_return": 0.234,
      "sharpe_ratio": 2.67,
      "max_drawdown": 0.087,
      "win_rate": 0.683,
      "total_trades": 1247
    },
    "model_metrics": {
      "accuracy": 0.724,
      "precision": 0.689,
      "recall": 0.756,
      "f1_score": 0.721
    }
  },
  "created_at": "2025-08-18T15:30:45Z",
  "next_deployment": "2025-08-19T09:30:00Z"
}
```

#### Get Strategy Performance

**Endpoint**: `GET /strategies/{strategy_id}/performance`

**Description**: Retrieve detailed performance metrics for a trading strategy.

**Query Parameters**:
- `start_date` (string): Performance period start
- `end_date` (string): Performance period end
- `include_trades` (boolean): Include individual trade details

**Response**:
```json
{
  "strategy_id": "STRAT-LSTM-001",
  "performance_period": {
    "start_date": "2025-07-01",
    "end_date": "2025-08-18"
  },
  "performance_metrics": {
    "total_return": 0.187,
    "annual_return": 0.234,
    "volatility": 0.089,
    "sharpe_ratio": 2.84,
    "sortino_ratio": 3.21,
    "max_drawdown": 0.067,
    "calmar_ratio": 3.49,
    "win_rate": 0.673,
    "profit_factor": 2.34
  },
  "risk_metrics": {
    "var_95": 0.023,
    "var_99": 0.034,
    "expected_shortfall": 0.041,
    "beta": 0.67,
    "alpha": 0.156,
    "information_ratio": 1.89
  },
  "trading_statistics": {
    "total_trades": 89,
    "winning_trades": 60,
    "losing_trades": 29,
    "average_win": 0.0345,
    "average_loss": -0.0156,
    "largest_win": 0.0891,
    "largest_loss": -0.0234,
    "consecutive_wins": 8,
    "consecutive_losses": 3
  },
  "attribution_analysis": {
    "stock_selection": 0.156,
    "market_timing": 0.078,
    "sector_allocation": 0.023,
    "interaction_effects": -0.013
  }
}
```

### 2. Order Management

#### Place Order

**Endpoint**: `POST /orders`

**Description**: Place trading orders with advanced execution algorithms.

**Request Body**:
```json
{
  "symbol": "AAPL",
  "side": "BUY",
  "quantity": 1000,
  "order_type": "LIMIT",
  "price": 182.45,
  "time_in_force": "DAY",
  "execution_algorithm": {
    "type": "TWAP",
    "duration_minutes": 60,
    "participation_rate": 0.15,
    "price_improvement": true
  },
  "strategy_id": "STRAT-LSTM-001",
  "metadata": {
    "model_confidence": 0.847,
    "signal_strength": 0.734,
    "expected_return": 0.0234
  }
}
```

**Response** (201 Created):
```json
{
  "order_id": "ORD-2025-08-001234",
  "client_order_id": "STRAT-LSTM-001-20250818-001",
  "symbol": "AAPL",
  "side": "BUY",
  "quantity": 1000,
  "order_type": "LIMIT",
  "price": 182.45,
  "status": "PENDING_NEW",
  "time_in_force": "DAY",
  "created_at": "2025-08-18T15:30:45.123Z",
  "estimated_completion": "2025-08-18T16:30:45Z",
  "execution_estimate": {
    "expected_fill_price": 182.46,
    "expected_market_impact": 0.0001,
    "expected_slippage": 0.0005,
    "confidence_interval": [182.44, 182.48]
  }
}
```

#### Get Order Status

**Endpoint**: `GET /orders/{order_id}`

**Response**:
```json
{
  "order_id": "ORD-2025-08-001234",
  "symbol": "AAPL", 
  "side": "BUY",
  "quantity": 1000,
  "filled_quantity": 750,
  "remaining_quantity": 250,
  "avg_fill_price": 182.47,
  "status": "PARTIALLY_FILLED",
  "fills": [
    {
      "fill_id": "FILL-001",
      "quantity": 300,
      "price": 182.46,
      "timestamp": "2025-08-18T15:31:12.456Z",
      "venue": "NASDAQ"
    },
    {
      "fill_id": "FILL-002", 
      "quantity": 450,
      "price": 182.48,
      "timestamp": "2025-08-18T15:32:34.789Z",
      "venue": "NYSE"
    }
  ],
  "execution_metrics": {
    "implementation_shortfall": -0.0002,
    "market_impact": 0.0001,
    "timing_cost": 0.0001,
    "fill_rate": 0.75
  }
}
```

### 3. Portfolio Management

#### Get Portfolio Positions

**Endpoint**: `GET /portfolio/positions`

**Description**: Retrieve current portfolio positions with real-time P&L.

**Query Parameters**:
- `include_analytics` (boolean): Include advanced analytics
- `currency` (string): Base currency for reporting

**Response**:
```json
{
  "portfolio_id": "PORT-QUANT-001",
  "as_of_timestamp": "2025-08-18T15:30:45Z",
  "base_currency": "USD",
  "positions": [
    {
      "symbol": "AAPL",
      "quantity": 5000,
      "average_cost": 180.25,
      "current_price": 182.45,
      "market_value": 912250.00,
      "unrealized_pnl": 11000.00,
      "unrealized_pnl_percent": 0.0122,
      "realized_pnl": 2350.00,
      "total_pnl": 13350.00,
      "position_date": "2025-08-15",
      "weight": 0.0456,
      "beta": 1.23,
      "sector": "Technology",
      "analytics": {
        "var_contribution": 0.023,
        "correlation_to_portfolio": 0.67,
        "expected_return_1d": 0.0012,
        "volatility_contribution": 0.034
      }
    }
  ],
  "portfolio_summary": {
    "total_market_value": 20000000.00,
    "total_unrealized_pnl": 234567.89,
    "total_realized_pnl": 123456.78,
    "cash": 1500000.00,
    "margin_used": 500000.00,
    "buying_power": 2000000.00,
    "day_pnl": 45678.90,
    "portfolio_beta": 0.89,
    "number_of_positions": 25
  }
}
```

#### Portfolio Optimization

**Endpoint**: `POST /portfolio/optimize`

**Description**: Optimize portfolio allocation using modern portfolio theory and ML insights.

**Request Body**:
```json
{
  "portfolio_id": "PORT-QUANT-001",
  "optimization_objective": "MAXIMIZE_SHARPE_RATIO",
  "constraints": {
    "max_position_weight": 0.10,
    "min_position_weight": 0.01,
    "max_sector_concentration": 0.30,
    "target_beta": 1.0,
    "beta_tolerance": 0.2,
    "max_turnover": 0.25,
    "trading_cost_threshold": 0.001
  },
  "risk_model": "FAMA_FRENCH_5_FACTOR",
  "expected_returns_model": "ML_ENSEMBLE",
  "time_horizon_days": 30,
  "rebalance_frequency": "WEEKLY"
}
```

**Response**:
```json
{
  "optimization_id": "OPT-2025-08-001",
  "recommended_allocation": [
    {
      "symbol": "AAPL",
      "current_weight": 0.0456,
      "target_weight": 0.0523,
      "weight_change": 0.0067,
      "shares_to_trade": 342,
      "trade_direction": "BUY"
    },
    {
      "symbol": "GOOGL",
      "current_weight": 0.0389,
      "target_weight": 0.0298,
      "weight_change": -0.0091,
      "shares_to_trade": -64,
      "trade_direction": "SELL"
    }
  ],
  "optimization_results": {
    "expected_return": 0.12,
    "expected_volatility": 0.08,
    "expected_sharpe_ratio": 1.5,
    "tracking_error": 0.02,
    "turnover": 0.15,
    "estimated_trading_costs": 0.0008
  },
  "risk_attribution": {
    "factor_risk": 0.75,
    "specific_risk": 0.25,
    "top_risk_factors": [
      {"factor": "Market", "contribution": 0.45},
      {"factor": "Technology", "contribution": 0.18},
      {"factor": "Size", "contribution": 0.12}
    ]
  }
}
```

## Machine Learning APIs

### 1. Model Management

#### Get Model Performance

**Endpoint**: `GET /ml/models/{model_id}/performance`

**Description**: Retrieve ML model performance metrics and drift analysis.

**Response**:
```json
{
  "model_id": "lstm_ensemble_v2.1",
  "model_type": "LSTM_ENSEMBLE",
  "deployment_date": "2025-07-15T09:30:00Z",
  "performance_metrics": {
    "accuracy": 0.724,
    "precision": 0.689,
    "recall": 0.756,
    "f1_score": 0.721,
    "auc_roc": 0.845,
    "sharpe_ratio": 2.84,
    "information_coefficient": 0.156
  },
  "drift_analysis": {
    "feature_drift_score": 0.034,
    "prediction_drift_score": 0.021,
    "distribution_shift": 0.012,
    "drift_status": "STABLE",
    "last_retrain_date": "2025-08-01"
  },
  "feature_importance": [
    {
      "feature": "price_momentum_5d",
      "importance": 0.234,
      "stability": 0.89
    },
    {
      "feature": "volume_momentum_10d",
      "importance": 0.187,
      "stability": 0.92
    },
    {
      "feature": "volatility_ratio",
      "importance": 0.156,
      "stability": 0.85
    }
  ],
  "prediction_distribution": {
    "mean": 0.0045,
    "std": 0.0234,
    "skewness": 0.12,
    "kurtosis": 2.34,
    "confidence_bands": {
      "95": [-0.0413, 0.0503],
      "99": [-0.0572, 0.0662]
    }
  }
}
```

#### Generate Predictions

**Endpoint**: `POST /ml/predict`

**Description**: Generate ML predictions for specified symbols.

**Request Body**:
```json
{
  "model_id": "lstm_ensemble_v2.1",
  "symbols": ["AAPL", "GOOGL", "MSFT"],
  "prediction_horizon": [1, 5, 10],
  "include_confidence": true,
  "include_explanations": true
}
```

**Response**:
```json
{
  "prediction_id": "PRED-2025-08-001234",
  "model_id": "lstm_ensemble_v2.1",
  "timestamp": "2025-08-18T15:30:45Z",
  "predictions": [
    {
      "symbol": "AAPL",
      "current_price": 182.45,
      "predictions": {
        "1_day": {
          "return": 0.0123,
          "price": 184.69,
          "confidence": 0.847,
          "confidence_interval": [183.45, 185.93]
        },
        "5_day": {
          "return": 0.0387,
          "price": 189.51,
          "confidence": 0.723,
          "confidence_interval": [186.34, 192.68]
        },
        "10_day": {
          "return": 0.0612,
          "price": 193.62,
          "confidence": 0.645,
          "confidence_interval": [188.45, 198.79]
        }
      },
      "explanations": {
        "primary_drivers": [
          {
            "feature": "momentum_5d",
            "contribution": 0.0067,
            "description": "Strong 5-day momentum signal"
          },
          {
            "feature": "volume_anomaly",
            "contribution": 0.0034,
            "description": "Unusual volume pattern detected"
          }
        ],
        "risk_factors": [
          {
            "factor": "earnings_volatility",
            "impact": -0.0012,
            "description": "Elevated volatility around earnings"
          }
        ]
      }
    }
  ],
  "model_confidence": 0.762,
  "market_regime": "TRENDING",
  "processing_time_ms": 15.7
}
```

### 2. Feature Engineering

#### Get Feature Data

**Endpoint**: `GET /ml/features`

**Description**: Retrieve engineered features for model training and analysis.

**Query Parameters**:
- `symbols` (string): Comma-separated symbols
- `features` (string): Comma-separated feature names
- `start_date` (string): Start date for feature data
- `end_date` (string): End date for feature data

**Response**:
```json
{
  "features": [
    {
      "symbol": "AAPL",
      "timestamp": "2025-08-18T15:30:00Z",
      "features": {
        "price_momentum_5d": 0.0234,
        "volume_momentum_10d": 0.0456,
        "volatility_ratio": 1.23,
        "rsi_divergence": -0.12,
        "macd_signal": 0.045,
        "bollinger_position": 0.67,
        "volume_price_trend": 0.034,
        "williams_r": -23.4,
        "stochastic_k": 67.8,
        "correlation_spy": 0.78
      }
    }
  ],
  "feature_metadata": {
    "total_features": 47,
    "feature_groups": ["momentum", "volatility", "volume", "sentiment"],
    "update_frequency": "REAL_TIME",
    "last_updated": "2025-08-18T15:30:45Z"
  }
}
```

## Risk Management APIs

### 1. Real-Time Risk Monitoring

#### Get Portfolio Risk Metrics

**Endpoint**: `GET /risk/portfolio/{portfolio_id}`

**Description**: Real-time portfolio risk analysis with comprehensive metrics.

**Response**:
```json
{
  "portfolio_id": "PORT-QUANT-001",
  "risk_timestamp": "2025-08-18T15:30:45Z",
  "value_at_risk": {
    "var_1d_95": 234567.89,
    "var_1d_99": 345678.90,
    "var_10d_95": 567890.12,
    "expected_shortfall_95": 456789.01,
    "confidence_level": 0.95
  },
  "portfolio_analytics": {
    "total_exposure": 20000000.00,
    "net_exposure": 18500000.00,
    "gross_exposure": 21500000.00,
    "portfolio_beta": 0.89,
    "correlation_to_market": 0.78,
    "concentration_risk": 0.23,
    "leverage_ratio": 1.075
  },
  "sector_exposure": {
    "Technology": 0.35,
    "Healthcare": 0.18,
    "Financial": 0.15,
    "Consumer": 0.12,
    "Industrial": 0.10,
    "Energy": 0.05,
    "Other": 0.05
  },
  "risk_factors": [
    {
      "factor": "Market Risk",
      "exposure": 0.67,
      "contribution_to_var": 0.45,
      "sensitivity": 1.23
    },
    {
      "factor": "Technology Sector",
      "exposure": 0.35,
      "contribution_to_var": 0.28,
      "sensitivity": 1.45
    }
  ],
  "stress_test_results": [
    {
      "scenario": "2008_FINANCIAL_CRISIS",
      "portfolio_loss": -0.23,
      "confidence": 0.85
    },
    {
      "scenario": "COVID_MARCH_2020",
      "portfolio_loss": -0.18,
      "confidence": 0.89
    }
  ]
}
```

#### Risk Alerts

**Endpoint**: `GET /risk/alerts`

**Description**: Active risk alerts and limit breaches.

**Response**:
```json
{
  "alerts": [
    {
      "alert_id": "ALERT-2025-08-001",
      "severity": "HIGH",
      "type": "POSITION_LIMIT_BREACH",
      "symbol": "AAPL",
      "current_value": 0.0567,
      "limit_value": 0.05,
      "breach_amount": 0.0067,
      "timestamp": "2025-08-18T15:25:12Z",
      "action_required": "REDUCE_POSITION"
    },
    {
      "alert_id": "ALERT-2025-08-002",
      "severity": "MEDIUM",
      "type": "CORRELATION_ANOMALY",
      "description": "Unusual correlation between AAPL and GOOGL detected",
      "current_correlation": 0.23,
      "historical_correlation": 0.67,
      "z_score": -3.45,
      "timestamp": "2025-08-18T15:20:45Z"
    }
  ],
  "alert_summary": {
    "total_alerts": 2,
    "high_severity": 1,
    "medium_severity": 1,
    "low_severity": 0
  }
}
```

### 2. Stress Testing

#### Run Stress Test

**Endpoint**: `POST /risk/stress-test`

**Request Body**:
```json
{
  "portfolio_id": "PORT-QUANT-001",
  "scenario_type": "CUSTOM",
  "stress_parameters": {
    "market_shock": -0.15,
    "volatility_spike": 2.0,
    "correlation_breakdown": true,
    "liquidity_crisis": false
  },
  "time_horizon_days": 10,
  "confidence_levels": [0.95, 0.99, 0.999]
}
```

**Response**:
```json
{
  "stress_test_id": "STRESS-2025-08-001",
  "portfolio_id": "PORT-QUANT-001",
  "scenario_type": "CUSTOM",
  "results": {
    "portfolio_loss": -0.127,
    "portfolio_loss_amount": -2540000.00,
    "worst_case_loss": -0.189,
    "probability_of_loss": 0.23,
    "expected_shortfall": -0.156
  },
  "position_impact": [
    {
      "symbol": "AAPL",
      "current_exposure": 912250.00,
      "stressed_value": 756834.50,
      "loss_amount": -155415.50,
      "loss_percentage": -0.17
    }
  ],
  "risk_decomposition": {
    "market_risk": 0.67,
    "idiosyncratic_risk": 0.23,
    "interaction_effects": 0.10
  }
}
```

## Analytics & Reporting APIs

### 1. Performance Analytics

#### Generate Performance Report

**Endpoint**: `POST /analytics/performance-report`

**Request Body**:
```json
{
  "portfolio_id": "PORT-QUANT-001",
  "report_type": "COMPREHENSIVE",
  "start_date": "2025-07-01",
  "end_date": "2025-08-18",
  "benchmark": "SPY",
  "attribution_model": "BRINSON_FACHLER",
  "risk_model": "FAMA_FRENCH_5_FACTOR"
}
```

**Response**:
```json
{
  "report_id": "RPT-2025-08-001",
  "performance_summary": {
    "total_return": 0.187,
    "benchmark_return": 0.124,
    "excess_return": 0.063,
    "tracking_error": 0.045,
    "information_ratio": 1.40,
    "alpha": 0.068,
    "beta": 0.89
  },
  "attribution_analysis": {
    "security_selection": 0.043,
    "asset_allocation": 0.012,
    "interaction_effect": 0.008,
    "total_active_return": 0.063
  },
  "risk_analysis": {
    "portfolio_volatility": 0.089,
    "benchmark_volatility": 0.097,
    "correlation": 0.87,
    "max_drawdown": -0.067,
    "var_95": 0.023
  },
  "url": "https://reports.quantrading.enterprise.com/RPT-2025-08-001.pdf"
}
```

## Data Models

### Symbol Configuration Schema

```json
{
  "type": "object",
  "properties": {
    "symbol": {"type": "string", "pattern": "^[A-Z]{1,5}$"},
    "exchange": {"type": "string", "enum": ["NYSE", "NASDAQ", "AMEX"]},
    "asset_type": {"type": "string", "enum": ["STOCK", "ETF", "OPTION", "FUTURE"]},
    "currency": {"type": "string", "enum": ["USD", "EUR", "GBP", "JPY"]},
    "sector": {"type": "string"},
    "market_cap": {"type": "number", "minimum": 0},
    "average_volume": {"type": "number", "minimum": 0},
    "trading_enabled": {"type": "boolean"},
    "min_quantity": {"type": "number", "minimum": 1},
    "max_quantity": {"type": "number"},
    "tick_size": {"type": "number", "minimum": 0.01}
  },
  "required": ["symbol", "exchange", "asset_type", "currency"]
}
```

### Order Schema

```json
{
  "type": "object",
  "properties": {
    "symbol": {"type": "string"},
    "side": {"type": "string", "enum": ["BUY", "SELL"]},
    "quantity": {"type": "number", "minimum": 1},
    "order_type": {"type": "string", "enum": ["MARKET", "LIMIT", "STOP", "STOP_LIMIT"]},
    "price": {"type": "number", "minimum": 0},
    "stop_price": {"type": "number", "minimum": 0},
    "time_in_force": {"type": "string", "enum": ["DAY", "GTC", "IOC", "FOK"]},
    "strategy_id": {"type": "string"},
    "execution_algorithm": {
      "type": "object",
      "properties": {
        "type": {"type": "string", "enum": ["TWAP", "VWAP", "POV", "IS"]},
        "duration_minutes": {"type": "number", "minimum": 1},
        "participation_rate": {"type": "number", "minimum": 0, "maximum": 1}
      }
    }
  },
  "required": ["symbol", "side", "quantity", "order_type"]
}
```

## Error Handling

### HTTP Status Codes

| Code | Description | Usage |
|------|-------------|-------|
| 200 | OK | Successful request |
| 201 | Created | Resource created successfully |
| 202 | Accepted | Request accepted for processing |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Authentication required |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 409 | Conflict | Resource conflict (duplicate order) |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Market closed or system maintenance |

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_ORDER",
    "message": "Order quantity exceeds position limit",
    "details": {
      "symbol": "AAPL",
      "requested_quantity": 10000,
      "max_allowed": 5000,
      "current_position": 3000
    }
  },
  "request_id": "REQ-2025-08-001234",
  "timestamp": "2025-08-18T15:30:45Z"
}
```

## Rate Limiting & Performance

### Rate Limits
- **Market Data**: 50,000 requests/minute
- **Trading**: 10,000 requests/minute  
- **Analytics**: 1,000 requests/minute
- **WebSocket**: 1,000 subscriptions per connection

### Performance SLAs
- **Market Data Latency**: <1ms (99th percentile)
- **Order Acknowledgment**: <5ms (99th percentile)
- **Portfolio Updates**: <10ms (99th percentile)
- **Analytics Queries**: <100ms (95th percentile)

## WebSocket Channels

### Available Channels
- `quotes`: Real-time price quotes
- `trades`: Trade executions
- `orders`: Order status updates
- `portfolio`: Portfolio position changes
- `risk`: Risk alerts and metrics
- `market_data`: Level 2 market data

### Authentication
```json
{
  "action": "auth",
  "token": "your_access_token",
  "api_key": "your_api_key"
}
```

## SDK Examples

### Python SDK

```python
from quantrading_api import QuantTradingClient

# Initialize client
client = QuantTradingClient(
    api_key="your_api_key",
    access_token="your_access_token",
    base_url="https://api.quantrading.enterprise.com/v2"
)

# Get real-time quotes
quotes = client.get_quotes(["AAPL", "GOOGL", "MSFT"])

# Place order
order = client.place_order(
    symbol="AAPL",
    side="BUY", 
    quantity=1000,
    order_type="LIMIT",
    price=182.45
)

# Get portfolio performance
performance = client.get_portfolio_performance("PORT-QUANT-001")
```

### JavaScript SDK

```javascript
const QuantTradingAPI = require('@quantrading/api-client');

const client = new QuantTradingAPI({
  apiKey: 'your_api_key',
  accessToken: 'your_access_token',
  baseURL: 'https://api.quantrading.enterprise.com/v2'
});

// Subscribe to real-time data
const ws = client.createWebSocket();
ws.subscribe('quotes', ['AAPL', 'GOOGL']);

// Place order
const order = await client.placeOrder({
  symbol: 'AAPL',
  side: 'BUY',
  quantity: 1000,
  orderType: 'LIMIT',
  price: 182.45
});
```

## Support & Resources

- **API Status**: https://status.quantrading.enterprise.com
- **Developer Portal**: https://developers.quantrading.enterprise.com
- **Documentation**: https://docs.quantrading.enterprise.com
- **Support**: api-support@quantrading.enterprise.com
- **Sandbox Environment**: https://sandbox.quantrading.enterprise.com

---

This API documentation provides comprehensive guidance for integrating with the Quantitative Trading Intelligence System, enabling institutional-grade algorithmic trading with advanced ML capabilities and real-time risk management.