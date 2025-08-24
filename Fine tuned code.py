# Standard library imports
import os
import sys
import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

# Data manipulation and analysis
import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy import stats
from statsmodels.tsa.stattools import adfuller

# Trading and financial APIs
import MetaTrader5 as mt5
import yfinance as yf
from anthropic import Anthropic

# Time zone handling
import pytz
from zoneinfo import ZoneInfo

# Technical analysis
import talib
from finta import TA

# Configuration
from dotenv import load_dotenv

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)

class OrderType(Enum):
    BUY = mt5.ORDER_TYPE_BUY
    SELL = mt5.ORDER_TYPE_SELL
    BUY_LIMIT = mt5.ORDER_TYPE_BUY_LIMIT
    SELL_LIMIT = mt5.ORDER_TYPE_SELL_LIMIT
    BUY_STOP = mt5.ORDER_TYPE_BUY_STOP
    SELL_STOP = mt5.ORDER_TYPE_SELL_STOP

class Position:
    def __init__(self, ticket: int, symbol: str, type: OrderType, volume: float, 
                 price_open: float, sl: float, tp: float):
        self.ticket = ticket
        self.symbol = symbol
        self.type = type
        self.volume = volume
        self.price_open = price_open
        self.sl = sl
        self.tp = tp

class TechnicalAnalyzer:
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['SMA_200'] = df['close'].rolling(window=200).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2 * df['close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2 * df['close'].rolling(window=20).std()
        
        return df

class TradingSystem:
    def __init__(self, username: int, password: str, server: str, path: str = None):
        """Initialize MT5 connection"""
        if not mt5.initialize(path=path):
            raise Exception(f"MT5 initialization failed: {mt5.last_error()}")
        
        if not mt5.login(username, password, server):
            mt5.shutdown()
            raise Exception(f"MT5 login failed: {mt5.last_error()}")
        
        self.timezone = pytz.timezone("Etc/UTC")
        logger.info("MT5 Trading System initialized successfully")
        
    def get_account_info(self) -> Dict:
        """Get account information"""
        account_info = mt5.account_info()
        if account_info is None:
            raise Exception(f"Failed to get account info: {mt5.last_error()}")
        
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free,
            'leverage': account_info.leverage
        }

    def get_historical_data(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        """Get historical price data from MT5"""
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        
        rates = mt5.copy_rates_from_pos(symbol, timeframe_map[timeframe], 0, bars)
        if rates is None:
            raise Exception(f"Failed to get historical data: {mt5.last_error()}")
        
        df = pd.DataFrame(rates)
        df = TechnicalAnalyzer.calculate_indicators(df)
        return df

    def place_order(self, symbol: str, order_type: OrderType, volume: float,
                   price: float = None, sl_points: int = None, tp_points: int = None) -> bool:
        """Place a market or pending order"""
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise Exception(f"Symbol {symbol} not found")
        
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                raise Exception(f"Symbol {symbol} selection failed")

        point = symbol_info.point
        
        if order_type in [OrderType.BUY, OrderType.SELL]:
            price = mt5.symbol_info_tick(symbol).ask if order_type == OrderType.BUY else mt5.symbol_info_tick(symbol).bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL if order_type in [OrderType.BUY, OrderType.SELL] else mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": volume,
            "type": order_type.value,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": f"python_{order_type.name.lower()}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if sl_points:
            request["sl"] = price - sl_points * point if order_type in [OrderType.BUY, OrderType.BUY_LIMIT, OrderType.BUY_STOP] else price + sl_points * point
        if tp_points:
            request["tp"] = price + tp_points * point if order_type in [OrderType.BUY, OrderType.BUY_LIMIT, OrderType.BUY_STOP] else price - tp_points * point

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.comment}")
            return False
        
        logger.info(f"Order placed successfully: {result.order}")
        return True

class QuarterlyReportAnalyzer:
    def __init__(self, anthropic_api_key: str, mt5_username: int = None,
                 mt5_password: str = None, mt5_server: str = None):
        self.anthropic = Anthropic(api_key=anthropic_api_key)
        self.trading_system = None
        
        if all([mt5_username, mt5_password, mt5_server]):
            self.trading_system = TradingSystem(mt5_username, mt5_password, mt5_server)
        
        self.analysis_template = """
        Based on the following quarterly report data and technical analysis, provide a detailed analysis and recommendation:
        
        Financial Metrics:
        {metrics}
        
        Technical Analysis:
        {technical_analysis}
        
        Current Positions:
        {positions}
        
        Please provide:
        1. Financial health summary
        2. Technical analysis overview
        3. Key strengths and weaknesses
        4. Risk factors
        5. Trading recommendations:
           - For new positions: BUY/SELL/HOLD with entry price, stop loss, and take profit
           - For existing positions: HOLD/EXIT with reasoning
        """

    def fetch_financial_data(self, ticker: str) -> Dict:
        """Fetch financial data using yfinance"""
        stock = yf.Ticker(ticker)
        
        return {
            'financials': stock.quarterly_financials,
            'balance_sheet': stock.quarterly_balance_sheet,
            'cashflow': stock.quarterly_cashflow,
            'info': stock.info
        }

    def analyze_quarterly_report(self, symbol: str) -> Tuple[str, Dict]:
        """Analyze financial and technical data"""
        try:
            financial_data = self.fetch_financial_data(symbol)
            metrics = self.calculate_key_metrics(financial_data)
            
            df = self.trading_system.get_historical_data(symbol, "D1", 200)
            technical_analysis = self.analyze_technical_data(df)
            
            positions = self.trading_system.get_open_positions(symbol)
            positions_text = "\n".join([
                f"Position {p.ticket}: {p.type.name}, Volume: {p.volume}"
                for p in positions
            ])
            
            message = self.anthropic.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1500,
                temperature=0.1,
                system="You are a professional financial analyst providing trading recommendations.",
                messages=[{
                    "role": "user",
                    "content": self.analysis_template.format(
                        metrics=json.dumps(metrics, indent=2),
                        technical_analysis=json.dumps(technical_analysis, indent=2),
                        positions=positions_text
                    )
                }]
            )
            
            return message.content, metrics

        except Exception as e:
            logger.error(f"Analysis error for {symbol}: {str(e)}")
            return str(e), {}

    def execute_trades(self, symbol: str, analysis: str):
        """Execute trades based on analysis"""
        try:
            if "BUY" in analysis.upper():
                volume = self.calculate_position_size(symbol)
                self.trading_system.place_order(
                    symbol=symbol,
                    order_type=OrderType.BUY,
                    volume=volume,
                    sl_points=100,
                    tp_points=300
                )
                logger.info(f"Buy order executed for {symbol}")
                
            elif "SELL" in analysis.upper():
                volume = self.calculate_position_size(symbol)
                self.trading_system.place_order(
                    symbol=symbol,
                    order_type=OrderType.SELL,
                    volume=volume,
                    sl_points=100,
                    tp_points=300
                )
                logger.info(f"Sell order executed for {symbol}")
                
            if "EXIT" in analysis.upper():
                positions = self.trading_system.get_open_positions(symbol)
                for position in positions:
                    self.trading_system.close_position(position)
                    logger.info(f"Closed position {position.ticket} for {symbol}")
                    
        except Exception as e:
            logger.error(f"Trade execution error: {str(e)}")

    def calculate_position_size(self, symbol: str, risk_percentage: float = 0.02) -> float:
        """Calculate position size based on risk management"""
        account_info = self.trading_system.get_account_info()
        risk_amount = account_info['balance'] * risk_percentage
        
        symbol_info = mt5.symbol_info(symbol)
        point_value = 100 * symbol_info.point
        volume = risk_amount / point_value
        
        return max(symbol_info.volume_min, min(round(volume, 2), symbol_info.volume_max))

def main():
    try:
        # Load configuration
        api_key = os.getenv('ANTHROPIC_API_KEY')
        mt5_username = int(os.getenv('MT5_USERNAME'))
        mt5_password = os.getenv('MT5_PASSWORD')
        mt5_server = os.getenv('MT5_SERVER')
        
        # Initialize analyzer
        analyzer = QuarterlyReportAnalyzer(
            anthropic_api_key=api_key,
            mt5_username=mt5_username,
            mt5_password=mt5_password,
            mt5_server=mt5_server
        )
        
        symbols = ['AAPL', 'MSFT', 'GOOGL']  # Add your symbols here
        
        while True:
            try:
                for symbol in symbols:
                    logger.info(f"Analyzing {symbol}")
                    
                    # Perform analysis
                    analysis, metrics = analyzer.analyze_quarterly_report(symbol)
                    
                    # Execute trades based on analysis
                    analyzer.execute_trades(symbol, analysis)
                    
                    # Save analysis to file
                    with open(f"analysis_{symbol}_{datetime.now().strftime('%Y%m%d')}.txt", 'w') as f:
                        f.write(analysis)
                    
                # Wait before next analysis
                time.sleep(3600)  # 1 hour interval
                
            except KeyboardInterrupt:
                logger.info("Stopping the trading system...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                time.sleep(60)  # Wait 1 minute before retrying
                
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
