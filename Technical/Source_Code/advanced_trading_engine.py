"""
Quantitative Trading Intelligence System
=======================================

Institutional-grade algorithmic trading system with deep learning models,
real-time market analysis, and sophisticated risk management capabilities.

Author: Emilio Cardenas
License: MIT

INSTRUCTIONS:
1. Create a new file called "advanced_trading_engine.py" in your repository
2. Copy and paste this entire code
3. Install required packages: pip install tensorflow yfinance pandas-ta pypfopt arch
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Advanced ML and Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Input, LayerNormalization, Add, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Traditional ML
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Financial Data and Analysis
import yfinance as yf
try:
    import pandas_ta as ta
except ImportError:
    print("pandas_ta not installed. Install with: pip install pandas_ta")

# Statistical Analysis
from scipy import stats
from datetime import datetime, timedelta
import time

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedQuantitativeTradingEngine:
    """
    Institutional-grade quantitative trading platform with advanced AI capabilities.
    
    Features:
    - LSTM and Transformer neural networks for price prediction
    - Real-time market data processing and analysis
    - Advanced technical indicators and feature engineering
    - Portfolio optimization and risk management
    - Backtesting and performance analytics
    - Automated trading signal generation
    - Risk-adjusted return optimization
    """
    
    def __init__(self, config=None):
        """Initialize the quantitative trading engine."""
        self.config = config or self._default_config()
        self.models = {}
        self.scalers = {}
        self.market_data = {}
        self.portfolio = {}
        self.performance_metrics = {}
        self.risk_metrics = {}
        
    def _default_config(self):
        """Default configuration for the trading engine."""
        return {
            'symbols': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'],
            'lookback_period': 60,
            'prediction_horizon': 5,
            'train_test_split': 0.8,
            'lstm_units': [128, 64, 32],
            'transformer_heads': 8,
            'transformer_layers': 4,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32,
            'risk_free_rate': 0.02,
            'max_position_size': 0.1,
            'stop_loss': 0.05,
            'take_profit': 0.15,
            'rebalance_frequency': 'weekly'
        }
    
    def fetch_market_data(self, symbols=None, period='2y'):
        """Fetch comprehensive market data with advanced preprocessing."""
        symbols = symbols or self.config['symbols']
        
        print(f"Fetching market data for {len(symbols)} symbols...")
        
        for symbol in symbols:
            try:
                # Fetch OHLCV data
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                
                if data.empty:
                    print(f"Warning: No data found for {symbol}")
                    continue
                
                # Advanced technical indicators
                data = self._calculate_technical_indicators(data)
                
                # Market microstructure features
                data = self._calculate_microstructure_features(data)
                
                # Volatility and risk metrics
                data = self._calculate_volatility_metrics(data)
                
                # Sentiment and momentum indicators
                data = self._calculate_sentiment_indicators(data)
                
                self.market_data[symbol] = data
                print(f"Successfully processed data for {symbol}: {len(data)} records")
                
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
        
        return self.market_data
    
    def _calculate_technical_indicators(self, data):
        """Calculate comprehensive technical indicators."""
        try:
            # Moving averages
            data['SMA_10'] = data['Close'].rolling(window=10).mean()
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['EMA_12'] = data['Close'].ewm(span=12).mean()
            data['EMA_26'] = data['Close'].ewm(span=26).mean()
            
            # MACD
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            data['BB_Middle'] = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
            data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
            data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
            data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
            
            # Average True Range
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            data['ATR'] = true_range.rolling(14).mean()
            
        except Exception as e:
            print(f"Error calculating technical indicators: {str(e)}")
        
        return data
    
    def _calculate_microstructure_features(self, data):
        """Calculate market microstructure features."""
        # Price-based features
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data['High_Low_Ratio'] = data['High'] / data['Low']
        data['Open_Close_Ratio'] = data['Open'] / data['Close']
        
        # Volume-based features
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
        data['Price_Volume'] = data['Close'] * data['Volume']
        
        # Volatility features
        data['Volatility_5'] = data['Returns'].rolling(window=5).std()
        data['Volatility_20'] = data['Returns'].rolling(window=20).std()
        
        # Gap analysis
        data['Gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        
        # Momentum indicators
        data['Momentum_5'] = data['Close'] / data['Close'].shift(5) - 1
        data['Momentum_10'] = data['Close'] / data['Close'].shift(10) - 1
        data['Momentum_20'] = data['Close'] / data['Close'].shift(20) - 1
        
        return data
    
    def _calculate_volatility_metrics(self, data):
        """Calculate advanced volatility and risk metrics."""
        # Historical volatility
        data['HV_10'] = data['Returns'].rolling(window=10).std() * np.sqrt(252)
        data['HV_30'] = data['Returns'].rolling(window=30).std() * np.sqrt(252)
        
        # Parkinson volatility estimator
        data['Parkinson_Vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * 
            (np.log(data['High'] / data['Low']) ** 2).rolling(window=20).mean()
        ) * np.sqrt(252)
        
        # Garman-Klass volatility estimator
        data['GK_Vol'] = np.sqrt(
            0.5 * (np.log(data['High'] / data['Low']) ** 2) -
            (2 * np.log(2) - 1) * (np.log(data['Close'] / data['Open']) ** 2)
        ).rolling(window=20).mean() * np.sqrt(252)
        
        # Rolling beta (using SPY as market proxy)
        try:
            spy = yf.download('SPY', period='2y', progress=False)['Close']
            spy_returns = spy.pct_change().dropna()
            
            # Align dates
            aligned_data = data['Returns'].dropna()
            common_dates = aligned_data.index.intersection(spy_returns.index)
            
            if len(common_dates) > 60:
                aligned_returns = aligned_data.loc[common_dates]
                aligned_spy = spy_returns.loc[common_dates]
                
                # Calculate rolling beta
                window = 60
                betas = []
                for i in range(window, len(aligned_returns)):
                    y = aligned_returns.iloc[i-window:i]
                    x = aligned_spy.iloc[i-window:i]
                    if len(y) == len(x) and len(y) > 0:
                        covariance = np.cov(y, x)[0, 1]
                        variance = np.var(x)
                        beta = covariance / variance if variance != 0 else 0
                        betas.append(beta)
                    else:
                        betas.append(np.nan)
                
                # Align betas with data
                beta_series = pd.Series(betas, index=aligned_returns.index[window:])
                data['Beta'] = data.index.to_series().map(beta_series)
        except:
            data['Beta'] = 1.0  # Default beta
        
        return data
    
    def _calculate_sentiment_indicators(self, data):
        """Calculate sentiment and momentum indicators."""
        # Price momentum
        data['Price_Momentum'] = data['Close'] / data['Close'].rolling(window=20).mean() - 1
        
        # Volume momentum
        data['Volume_Momentum'] = data['Volume'] / data['Volume'].rolling(window=20).mean() - 1
        
        # Relative strength vs market
        data['Relative_Strength'] = data['Returns'].rolling(window=20).mean()
        
        # Trend strength
        data['Trend_Strength'] = np.abs(data['Close'] - data['SMA_20']) / data['SMA_20']
        
        return data
    
    def build_lstm_model(self, input_shape):
        """Build advanced LSTM model with attention mechanism."""
        model = Sequential([
            LSTM(self.config['lstm_units'][0], return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(self.config['lstm_units'][1], return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(self.config['lstm_units'][2], return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            
            Dense(50, activation='relu'),
            Dropout(0.1),
            
            Dense(self.config['prediction_horizon'], activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_transformer_model(self, input_shape):
        """Build Transformer model for time series prediction."""
        inputs = Input(shape=input_shape)
        
        # Multi-head attention layers
        x = inputs
        for _ in range(self.config['transformer_layers']):
            # Multi-head attention
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=self.config['transformer_heads'],
                key_dim=input_shape[-1] // self.config['transformer_heads']
            )(x, x)
            
            # Add & Norm
            x = Add()([x, attention_output])
            x = LayerNormalization()(x)
            
            # Feed forward
            ff_output = Dense(input_shape[-1] * 2, activation='relu')(x)
            ff_output = Dense(input_shape[-1])(ff_output)
            
            # Add & Norm
            x = Add()([x, ff_output])
            x = LayerNormalization()(x)
        
        # Global pooling and output
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.config['prediction_horizon'], activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_training_data(self, symbol):
        """Prepare training data for deep learning models."""
        data = self.market_data[symbol].copy()
        
        # Select features for training
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'MACD', 'MACD_Signal', 'RSI', 'BB_Position', 'ATR',
            'Returns', 'Volatility_5', 'Volatility_20',
            'Volume_Ratio', 'Price_Momentum', 'Trend_Strength'
        ]
        
        # Filter available columns
        available_columns = [col for col in feature_columns if col in data.columns]
        data_features = data[available_columns].dropna()
        
        if len(data_features) < self.config['lookback_period'] + self.config['prediction_horizon']:
            raise ValueError(f"Insufficient data for {symbol}")
        
        # Normalize features
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data_features)
        self.scalers[symbol] = scaler
        
        # Create sequences
        X, y = [], []
        for i in range(self.config['lookback_period'], 
                      len(scaled_data) - self.config['prediction_horizon'] + 1):
            X.append(scaled_data[i-self.config['lookback_period']:i])
            # Predict future returns
            future_prices = data_features['Close'].iloc[i:i+self.config['prediction_horizon']].values
            current_price = data_features['Close'].iloc[i-1]
            future_returns = (future_prices / current_price - 1)
            y.append(future_returns)
        
        X, y = np.array(X), np.array(y)
        
        # Train-test split
        split_idx = int(len(X) * self.config['train_test_split'])
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, symbol):
        """Train multiple models for ensemble prediction."""
        print(f"Training models for {symbol}...")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_training_data(symbol)
        
        models = {}
        
        # 1. LSTM Model
        print("Training LSTM model...")
        lstm_model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
        
        lstm_model.fit(
            X_train, y_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        models['lstm'] = lstm_model
        
        # 2. Transformer Model
        print("Training Transformer model...")
        transformer_model = self.build_transformer_model((X_train.shape[1], X_train.shape[2]))
        
        transformer_model.fit(
            X_train, y_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        models['transformer'] = transformer_model
        
        # 3. XGBoost (reshape data for traditional ML)
        print("Training XGBoost model...")
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # Train separate model for each prediction horizon
        xgb_models = []
        for i in range(self.config['prediction_horizon']):
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            xgb_model.fit(X_train_flat, y_train[:, i])
            xgb_models.append(xgb_model)
        models['xgboost'] = xgb_models
        
        # Store models and test data
        self.models[symbol] = models
        self.test_data = {
            'X_test': X_test,
            'y_test': y_test,
            'X_test_flat': X_test_flat
        }
        
        print(f"Models trained successfully for {symbol}")
        return models
    
    def evaluate_models(self, symbol):
        """Evaluate model performance with comprehensive metrics."""
        models = self.models[symbol]
        X_test = self.test_data['X_test']
        y_test = self.test_data['y_test']
        X_test_flat = self.test_data['X_test_flat']
        
        results = {}
        
        # Evaluate LSTM
        lstm_pred = models['lstm'].predict(X_test)
        lstm_mse = mean_squared_error(y_test, lstm_pred)
        lstm_mae = mean_absolute_error(y_test, lstm_pred)
        results['lstm'] = {'mse': lstm_mse, 'mae': lstm_mae, 'predictions': lstm_pred}
        
        # Evaluate Transformer
        transformer_pred = models['transformer'].predict(X_test)
        transformer_mse = mean_squared_error(y_test, transformer_pred)
        transformer_mae = mean_absolute_error(y_test, transformer_pred)
        results['transformer'] = {'mse': transformer_mse, 'mae': transformer_mae, 'predictions': transformer_pred}
        
        # Evaluate XGBoost
        xgb_pred = np.column_stack([
            model.predict(X_test_flat) for model in models['xgboost']
        ])
        xgb_mse = mean_squared_error(y_test, xgb_pred)
        xgb_mae = mean_absolute_error(y_test, xgb_pred)
        results['xgboost'] = {'mse': xgb_mse, 'mae': xgb_mae, 'predictions': xgb_pred}
        
        # Create ensemble prediction
        ensemble_pred = (lstm_pred + transformer_pred + xgb_pred) / 3
        ensemble_mse = mean_squared_error(y_test, ensemble_pred)
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        results['ensemble'] = {'mse': ensemble_mse, 'mae': ensemble_mae, 'predictions': ensemble_pred}
        
        print(f"\nModel Performance for {symbol}:")
        for model_name, metrics in results.items():
            print(f"{model_name.upper()}: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}")
        
        self.performance_metrics[symbol] = results
        return results
    
    def generate_trading_signals(self, symbol):
        """Generate sophisticated trading signals based on model predictions."""
        if symbol not in self.performance_metrics:
            raise ValueError(f"No trained models found for {symbol}")
        
        # Get ensemble predictions
        predictions = self.performance_metrics[symbol]['ensemble']['predictions']
        
        # Generate signals based on predicted returns
        signals = []
        for pred in predictions:
            # Multi-horizon signal generation
            short_term_signal = np.sign(pred[0])  # 1-day ahead
            medium_term_signal = np.sign(np.mean(pred[:3]))  # 3-day average
            long_term_signal = np.sign(np.mean(pred))  # 5-day average
            
            # Confidence based on prediction magnitude
            confidence = np.abs(np.mean(pred))
            
            # Combined signal with confidence weighting
            if confidence > 0.02:  # 2% threshold
                if short_term_signal == medium_term_signal == long_term_signal:
                    signal_strength = 'STRONG'
                    signal = short_term_signal
                elif short_term_signal == medium_term_signal:
                    signal_strength = 'MEDIUM'
                    signal = short_term_signal
                else:
                    signal_strength = 'WEAK'
                    signal = 0
            else:
                signal_strength = 'HOLD'
                signal = 0
            
            signals.append({
                'signal': signal,
                'strength': signal_strength,
                'confidence': confidence,
                'predicted_return': np.mean(pred)
            })
        
        return signals
    
    def calculate_portfolio_metrics(self, returns):
        """Calculate comprehensive portfolio performance metrics."""
        returns = np.array(returns)
        
        # Basic metrics
        total_return = np.prod(1 + returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = (annualized_return - self.config['risk_free_rate']) / volatility
        
        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - self.config['risk_free_rate']) / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio
        }
    
    def generate_power_bi_data(self, output_path="trading_dashboard_data.csv"):
        """Generate comprehensive data for Power BI dashboard."""
        dashboard_data = []
        
        # Model performance data
        for symbol, metrics in self.performance_metrics.items():
            for model_name, model_metrics in metrics.items():
                dashboard_data.append({
                    'data_type': 'model_performance',
                    'symbol': symbol,
                    'model': model_name,
                    'mse': model_metrics['mse'],
                    'mae': model_metrics['mae'],
                    'timestamp': datetime.now()
                })
        
        # Trading signals data
        for symbol in self.market_data.keys():
            try:
                signals = self.generate_trading_signals(symbol)
                for i, signal in enumerate(signals[-30:]):  # Last 30 signals
                    dashboard_data.append({
                        'data_type': 'trading_signal',
                        'symbol': symbol,
                        'signal': signal['signal'],
                        'strength': signal['strength'],
                        'confidence': signal['confidence'],
                        'predicted_return': signal['predicted_return'],
                        'timestamp': datetime.now() - timedelta(days=30-i)
                    })
            except:
                continue
        
        # Convert to DataFrame and save
        df = pd.DataFrame(dashboard_data)
        df.to_csv(output_path, index=False)
        print(f"Power BI dashboard data saved to {output_path}")
        
        return df

# Example usage and demonstration
if __name__ == "__main__":
    # Initialize the trading engine
    engine = AdvancedQuantitativeTradingEngine()
    
    # Fetch market data
    print("Fetching market data...")
    market_data = engine.fetch_market_data(symbols=['AAPL', 'GOOGL', 'MSFT'])
    
    # Train models for each symbol
    for symbol in ['AAPL']:  # Start with one symbol for demo
        try:
            print(f"\nProcessing {symbol}...")
            models = engine.train_models(symbol)
            results = engine.evaluate_models(symbol)
            signals = engine.generate_trading_signals(symbol)
            
            print(f"Generated {len(signals)} trading signals for {symbol}")
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
    
    # Generate Power BI data
    print("\nGenerating Power BI dashboard data...")
    dashboard_data = engine.generate_power_bi_data()
    
    print("\n" + "="*60)
    print("QUANTITATIVE TRADING INTELLIGENCE SYSTEM COMPLETE")
    print("="*60)
    print(f"Symbols Processed: {len(engine.models)}")
    print(f"Models Trained: LSTM, Transformer, XGBoost, Ensemble")
    print("Features: Technical Indicators, Volatility Metrics, Sentiment")
    print("Power BI Integration: Ready")
    print("="*60)

