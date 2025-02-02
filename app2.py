import yfinance as yf
import pandas as pd
import numpy as np
import itertools
import warnings
import json
import requests
import logging
import sys
import gc  # Import garbage collector
from tqdm import tqdm
from functools import reduce
from datetime import datetime
from alpaca_trade_api import REST
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

# Initialize Alpaca API
alpaca = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, API_BASE_URL)

api = tradeapi.REST(
    os.getenv('ALPACA_API_KEY'),
    os.getenv('ALPACA_SECRET_KEY'),
    'https://paper-api.alpaca.markets',
    api_version='v2'
)
# Define tickers in Alpaca syntax
# tickers = [
    #etfs
#     'QQQ', 'UUP', 'USO', 'XLV', 'XLP', 'XLE', 'XLF', 'XLI', 'XLB', 'XLK', 'XBI', 'XLU', 'XME', 'GDX', 'XOP',
#  'XHB', 'XRT', 'TLT', 'FDN', 'FEZ', 'DBA', 'IEF', 'DXJ', 'ITB', 'SPY', 'LQD', 'XLY', 'HYG', 'KWEB', 'FXI',
#  'EEM', 'DIA', 'IBB', 'IYT', 'EWZ', 'IWN', 'XLRE', 'URA', 'OIH', 'KBE', 'KRE', 'UVXY'
    
#   'SPY'
#   'BTCUSD', 'ETHUSD', 'LTCUSD', 'BCHUSD', 'SOLUSD', 'DOTUSD', 
#     'MATICUSD', 'LINKUSD', 'UNIUSD', 'DOGEUSD', 'AAVEUSD', 
#     'USDBTC', 'USDETH', 'USDLTC', 'USDBCH', 'USDSOL', 'USDDOT',

#     'ATOMUSD', 'AVAXUSD', 'ADAUSD', 'BNBUSD', 'SHIBUSD', 
#     'XLMUSD', 'XRPUSD', 'XTZUSD', 'ALGOUSD', 'MANAUSD', 'SANDUSD'
    
    
    # 'QQQ', 'UUP', 'USO', 'XLV', 'XLP', 'XLE', 'XLF', 'XLI', 'XLB', 'XLK', 'XBI', 'XLU', 'XME', 'GDX', 'XOP',
    #        'XHB', 'XRT', 'TLT', 'FDN', 'FEZ', 'DBA', 'IEF', 'DXJ', 'ITB', 'SPY', 'LQD', 'XLY', 'HYG', 'KWEB', 'FXI',
    #        'EEM', 'DIA', 'IBB', 'IYT', 'EWZ', 'IWN', 'XLRE', 'URA', 'OIH', 'KBE', 'KRE', 'AAPL', 'MSFT', 'AMZN',
    #        'GOOGL', 'META', 'NVDA', 'BRK-B', 'TSLA', 'UNH', 'JNJ', 'V', 'XOM', 'PG', 'MA', 'LLY', 'JPM', 'HD',
    #        'CVX', 'MRK', 'PEP', 'ABBV', 'KO', 'AVGO', 'COST', 'PFE', 'TMO', 'MCD', 'WMT', 'CSCO', 'ACN', 'ADBE', 'DHR',
    #        'LIN', 'NFLX', 'NEE', 'ABT', 'RTX', 'TXN', 'CRM', 'NKE', 'INTC', 'ORCL', 'MS', 'PM', 'MDT', 'SCHW', 'IBM',
    #        'HON', 'AMGN', 'LMT'
# ]

tickers = [
   'BTCUSD', 'ETHUSD', 'LTCUSD', 'BCHUSD', 'SOLUSD', 'DOTUSD', 
    'MATICUSD', 'LINKUSD', 'UNIUSD', 'DOGEUSD', 'AAVEUSD', 
    'USDBTC', 'USDETH', 'USDLTC', 'USDBCH', 'USDSOL', 'USDDOT',

    'ATOMUSD', 'AVAXUSD', 'ADAUSD', 'BNBUSD', 'SHIBUSD', 
    'XLMUSD', 'XRPUSD', 'XTZUSD', 'ALGOUSD', 'MANAUSD', 'SANDUSD'

  # Broad Market
  # 'QQQ', 'SPY', 'DIA', 'IWN', 'EEM', 'FXI', 'KWEB', 'EWZ', 'FEZ', 'ACWI',
  
  # # Sectors
  # 'XLV', 'XLP', 'XLE', 'XLF', 'XLK', 'XLI', 'XLB', 'XLY', 'XLU', 'XLRE',
  
  # # Industries
  # 'XBI', 'XME', 'GDX', 'XOP', 'URA', 'KBE', 'KRE', 'OIH', 'XHB', 'ITB', 'IYT',
  
  # # Commodities
  # 'USO', 'DBA', 'GLD', 'SLV', 'PICK', 'WOOD', 'REMX',
  
  # # Bonds and Fixed Income
  # 'TLT', 'IEF', 'LQD', 'HYG', 'TIP', 'BNDX',
  
  # # Thematic
  # 'FDN', 'ARKK', 'TAN', 'PBW', 'ROBO',
  
  # # Alternative Investments
  # 'PFF', 'PSP',
  
  # # Regional
  # 'VEA', 'VWO', 'SCZ', 'EWJ', 'INDA', 'EWA',
  
  # # Strategy
  # 'MTUM', 'QUAL', 'VLUE', 'RSP',
  
  # # Volatility
  # 'UVXY',
  
  # # Currency
  # 'UUP', 'CYB',
  
  # # Inverse ETFs (most liquid and paired with the above ETFs)
  # 'SQQQ', # ProShares UltraPro Short QQQ (pairs with QQQ)
  # 'SH',   # ProShares Short S&P500 (pairs with SPY)
  # 'DOG',  # ProShares Short Dow30 (pairs with DIA)
  # 'RWM',  # ProShares Short Russell2000 (pairs with IWN)
  # 'YXI',  # ProShares Short FTSE China 50 (pairs with FXI)
  # 'EDZ',  # Direxion Emerging Markets Bear 3x Shares (pairs with EEM)
  # 'DRV',  # Direxion Daily Real Estate Bear 3x Shares (pairs with XLRE)
  # 'ERY',  # Direxion Daily Energy Bear 3x Shares (pairs with XLE)
  # 'FAZ',  # Direxion Daily Financial Bear 3x Shares (pairs with XLF)
  # 'SOXS', # Direxion Daily Semiconductor Bear 3x Shares (pairs with XLK)
  # 'DUST', # Direxion Daily Gold Miners Bear 2x Shares (pairs with GDX)
  # 'SCO',  # ProShares UltraShort Bloomberg Crude Oil (pairs with USO)
  # 'TBT',  # ProShares UltraShort 20+ Year Treasury (pairs with TLT)
];

# Map tickers to yFinance syntax
# ticker_map = {ticker: ticker.replace("USD", "-USD") for ticker in tickers}
# ticker_map = {ticker: ticker.replace("", "") for ticker in tickers}
ticker_map = {ticker: ticker for ticker in tickers}


def fetch_data(tickers, period, interval):
    current_time = datetime.now()
    logging.info(f"Starting data fetch at {current_time}")
    data = {}
    
    # If single ticker passed as string, convert to list
    if isinstance(tickers, str):
        tickers = [tickers]
    
    for alpaca_ticker in tickers:
        try:
            yf_ticker = ticker_map.get(alpaca_ticker, alpaca_ticker)
            # Download data in smaller chunks if possible
            df = yf.download(yf_ticker, period=period, interval=interval)
            
            if not df.empty:
                # Log the time range of data
                logging.info(f"{alpaca_ticker} data range: {df.index[0]} to {df.index[-1]}")
                
                # Only keep necessary columns and convert to float32 for memory efficiency
                df = df[['Close']].astype('float32').copy()
                data[alpaca_ticker] = df
                print(f"Fetched data for {alpaca_ticker}")
            else:
                logging.warning(f"Empty data received for {alpaca_ticker}")
            
            # Explicitly delete the DataFrame to free memory
            del df
            gc.collect()  # Force garbage collection after each ticker
        except Exception as e:
            logging.error(f"Error fetching data for {alpaca_ticker}: {e}")
    return data


# The rest of the script continues as before, with no further adjustments needed for tickers

def calculate_indicators(df):
    if 'Close' not in df.columns:
        if 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        else:
            raise ValueError("Neither 'Close' nor 'Adj Close' column found in the DataFrame")

    df['daily_return'] = df['Close'].pct_change()
    df['volatility'] = df['daily_return'].rolling(window=2).std().rolling(window=89).mean()
    df['avg_return'] = df['daily_return'].rolling(window=89).mean()
    df['avg_volatility'] = df['volatility'].rolling(window=89).mean()
    df['rr_ratio'] = df['avg_return'] / df['avg_volatility']

    for n in [3, 13, 48, 200]:
        df[f'ROC_{n}'] = df['rr_ratio'].pct_change(n) * 100

    df['roc_compound'] = (df['ROC_3'] * 3 + df['ROC_13'] * 13 + df['ROC_48'] * 48 + df['ROC_200'] * 200)
    df['source_scaled'] = (df['rr_ratio'] - df['rr_ratio'].rolling(window=200).min()) / (
            df['rr_ratio'].rolling(window=200).max() - df['rr_ratio'].rolling(window=200).min())
    df['roc_compound_vs_price'] = df['roc_compound'] * df['source_scaled'].ewm(span=300, adjust=False).mean()
    df['diff_roc_compound_vs_price'] = df['roc_compound'] - df['roc_compound_vs_price']
    df['diff_roc_compound_vs_price_scaled'] = (df['diff_roc_compound_vs_price'] - df['diff_roc_compound_vs_price'].rolling(
        window=200).min()) / (df['diff_roc_compound_vs_price'].rolling(window=200).max() - df[
        'diff_roc_compound_vs_price'].rolling(window=200).min())
    df['diff_roc_compound_vs_price_scaled_vs_Closed_scaled'] = df['diff_roc_compound_vs_price_scaled'] - df[
        'source_scaled']
    df['differ'] = df['diff_roc_compound_vs_price_scaled_vs_Closed_scaled'] - df['source_scaled']

    df['single'] = (-1 * df['differ']).ewm(span=1, adjust=False).mean()
    df['single'] = (df['single'] * 2) - 1

    df['source_flip'] = 1 / df['Close']
    for n in [3, 13, 48, 200]:
        df[f'ROC_{n}_flip'] = df['source_flip'].pct_change(n) * 100

    df['roc_compound_flip'] = (df['ROC_3_flip'] * 3 + df['ROC_13_flip'] * 13 + df['ROC_48_flip'] * 48 + df['ROC_200_flip'] * 200)
    df['source_scaled_flip'] = (df['source_flip'] - df['source_flip'].rolling(window=200).min()) / (df['source_flip'].rolling(window=200).max() - df['source_flip'].rolling(window=200).min())
    df['roc_compound_vs_price_flip'] = df['roc_compound_flip'] * df['source_scaled_flip'].ewm(span=300, adjust=False).mean()
    df['diff_roc_compound_vs_price_flip'] = df['roc_compound_flip'] - df['roc_compound_vs_price_flip']
    df['diff_roc_compound_vs_price_scaled_flip'] = (df['diff_roc_compound_vs_price_flip'] - df['diff_roc_compound_vs_price_flip'].rolling(window=200).min()) / (df['diff_roc_compound_vs_price_flip'].rolling(window=200).max() - df['diff_roc_compound_vs_price_flip'].rolling(window=200).min())
    df['diff_roc_compound_vs_price_scaled_vs_Closed_scaled_flip'] = df['diff_roc_compound_vs_price_scaled_flip'] - df['source_scaled_flip']
    df['differ_flip'] = df['diff_roc_compound_vs_price_scaled_vs_Closed_scaled_flip'] - df['source_scaled_flip']

    df['single_flip'] = (-1 * df['differ_flip']).ewm(span=1, adjust=False).mean()
    df['single_flip'] = (df['single_flip'] * 2) - 1

    df['flip_diff'] = (df['single'] - df['single_flip']) / 2

    mult = np.abs(df['flip_diff'] - df['single'])
    ema_diff = np.abs(df['flip_diff'].ewm(span=27, adjust=False).mean() - df['single'].ewm(span=27, adjust=False).mean())
    avg_value = (df['flip_diff'] + df['single']) / 2
    df['score'] = avg_value - ((((avg_value * mult) * ema_diff) / (df['flip_diff'] + 5) + (df['single'] + 5) / 100) * ((df['flip_diff'] + 5 - (df['single'] + 5)) / 5))

    df['plot_bad'] = (((df['score'] - df['flip_diff']) * 2) + 1).ewm(span=1, adjust=False).mean()
    df['plot_good'] = (((df['score'] - df['flip_diff']) * 2) - 1).ewm(span=1, adjust=False).mean()

    return df

def calculate_ratio_indicators(df1, df2):
    ratio = df1['Close'] / df2['Close']
    df = pd.DataFrame(index=df1.index)
    df['Close'] = ratio
    return calculate_indicators(df)

def calculate_all_ratios(df_dict):
    ratio_dict = {}
    tickers = list(df_dict.keys())
    for ticker1, ticker2 in itertools.combinations(tickers, 2):
        try:
            ratio_dict[f"{ticker1}/{ticker2}"] = calculate_ratio_indicators(df_dict[ticker1], df_dict[ticker2])
        except Exception as e:
            logging.warning(f"Failed to calculate ratio for {ticker1}/{ticker2}: {e}")
    return ratio_dict

def find_best_long(df_dict_1m, df_dict_3m, df_dict_5m, ratio_dict_1m, ratio_dict_3m, ratio_dict_5m, date):
    threshold = -1.2
    candidates_1m = []
    candidates_3m = []
    candidates_5m = []
    
    # Find candidates for each timeframe
    for ticker, df in df_dict_1m.items():
        if date in df.index:
            current_score = df.loc[date, 'score']
            plot_good = df.loc[date, 'plot_good']
            
            if np.isscalar(current_score) and np.isscalar(plot_good):
                if current_score < threshold and abs(current_score - plot_good) < 0.2:
                    candidates_1m.append((ticker, current_score))

    for ticker, df in df_dict_3m.items():
        if date in df.index:
            current_score = df.loc[date, 'score']
            plot_good = df.loc[date, 'plot_good']
            
            if np.isscalar(current_score) and np.isscalar(plot_good):
                if current_score < threshold and abs(current_score - plot_good) < 0.2:
                    candidates_3m.append(ticker)

    for ticker, df in df_dict_5m.items():
        if date in df.index:
            current_score = df.loc[date, 'score']
            plot_good = df.loc[date, 'plot_good']
            
            if np.isscalar(current_score) and np.isscalar(plot_good):
                if current_score < threshold and abs(current_score - plot_good) < 0.2:
                    candidates_5m.append(ticker)

    # Only keep 1m candidates that appear in all timeframes
    filtered_candidates = [(ticker, score) for ticker, score in candidates_1m 
                         if ticker in candidates_3m and ticker in candidates_5m]

    if not filtered_candidates:
        return None

    filtered_candidates.sort(key=lambda x: x[1])
    top_candidates = filtered_candidates[:min(30, len(filtered_candidates))]

    best_long = None
    avg_ratio = float('inf')
    
    for ticker, _ in top_candidates:
        ratio_scores = []
        # Check ratios in all timeframes
        for ratio_dict in [ratio_dict_1m, ratio_dict_3m, ratio_dict_5m]:
            for ratio, ratio_df in ratio_dict.items():
                if ticker in ratio.split('/') and date in ratio_df.index:
                    ratio_score = ratio_df.loc[date, 'score']
                    ratio_plot_good = ratio_df.loc[date, 'plot_good']
                    
                    if np.isscalar(ratio_score) and np.isscalar(ratio_plot_good):
                        if ratio_score < -1 and abs(ratio_score - ratio_plot_good) < 0.2:
                            ratio_scores.append(ratio_score)

        if ratio_scores:
            current_avg = sum(ratio_scores) / len(ratio_scores)
            if current_avg < avg_ratio:
                avg_ratio = current_avg
                best_long = ticker
                
    return best_long

def fetch_account_cash():
    account = alpaca.get_account()
    cash = float(account.cash)
    logging.info(f"Current account cash: ${cash:.2f}")  # Log current account cash
    return cash

def is_market_open():
    """Check if the market is currently open"""
    clock = alpaca.get_clock()
    return clock.is_open

def cancel_existing_orders(symbol=None):
    """Cancel all existing orders, or orders for a specific symbol"""
    try:
        if symbol:
            orders = alpaca.list_orders(symbol=symbol, status='open')
            for order in orders:
                alpaca.cancel_order(order.id)
                logging.info(f"Cancelled existing order for {symbol}")
        else:
            alpaca.cancel_all_orders()
            logging.info("Cancelled all existing orders")
    except Exception as e:
        logging.error(f"Error cancelling orders: {e}")

def place_order(symbol, amount, side, quantity=None):
    """
    Place an order on Alpaca with stop loss and allow fractional shares.
    :param symbol: Ticker symbol (e.g., 'BTCUSD').
    :param amount: Dollar amount for 'buy' orders.
    :param side: 'buy' or 'sell'.
    :param quantity: Quantity for 'sell' orders.
    """
    if not is_market_open():
        logging.info("Market is closed. Skipping order placement.")
        return

    try:
        # Cancel any existing orders for this symbol first
        cancel_existing_orders(symbol)

        if side == 'buy':
            # Get current price
            last_quote = alpaca.get_latest_trade(symbol)
            price = float(last_quote.price)
            
            # Calculate the amount to invest with 2x leverage
            total_investment = amount * 2
            
            # Calculate the number of shares (allowing fractional shares)
            shares = total_investment / price
            
            if shares > 0:
                # Place the market order
                alpaca.submit_order(
                    symbol=symbol,
                    qty=shares,
                    side=side,
                    type='market',
                    time_in_force='day'  # Use 'day' to avoid order stacking
                )
                logging.info(f"Placed {side} order for {shares:.4f} shares of {symbol}")

                # Set a stop loss order at 3% below the entry price
                stop_loss_price = price * 0.97  # 3% stop loss
                alpaca.submit_order(
                    symbol=symbol,
                    qty=shares,
                    side='sell',
                    type='stop_market',
                    stop_price=stop_loss_price,
                    time_in_force='gtc'  # Good 'til canceled
                )
                logging.info(f"Placed stop loss order for {shares:.4f} shares of {symbol} at ${stop_loss_price:.2f}")
            else:
                logging.warning(f"Amount {amount} too small to buy shares of {symbol}")
        elif side == 'sell' and quantity:
            alpaca.submit_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                type='market',
                time_in_force='day'
            )
            logging.info(f"Placed {side} order for {quantity} units of {symbol}")
        else:
            logging.error("Invalid order parameters: qty or notional is required")
    except Exception as e:
        logging.error(f"Error placing {side} order for {symbol}: {e}")

max_positions = 20

total_cash = fetch_account_cash()  # Fetch cash at the start of the cycle
def calculate_position_size():
    allocation = (1/max_positions) * total_cash  # Allocate 5% of the total cash
    return round(allocation, 2)  # Return the dollar amount, rounded to 2 decimal places
# Allow fractional values, minimum position size of 0.01

def calculate_portfolio_proportions(long_positions, total_cash, df_dict):
    print("Calculating portfolio proportions")
    portfolio_values = []
    total_equity_value = 0

    for pos in long_positions:
        # Safeguard to ensure 'ticker' and 'quantity' are present and valid
        ticker = pos.get('ticker')
        quantity = pos.get('quantity')
        if ticker in df_dict and quantity:
            current_price = df_dict[ticker].iloc[-1]['Close']  # Use the latest price
            equity_value = quantity * current_price
            portfolio_values.append({'ticker': ticker, 'value': equity_value})
            total_equity_value += equity_value

    total_portfolio_value = total_equity_value + total_cash

    proportions = []
    for equity in portfolio_values:
        proportion = (equity['value'] / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0
        proportions.append({'ticker': equity['ticker'], 'proportion': proportion})

    return proportions

def log_portfolio_proportions(long_positions, total_cash, df_dict):
    print("Logging portfolio proportions")
    if not long_positions:
        logging.warning("No long positions to log proportions for.")
        return

    try:
        proportions = calculate_portfolio_proportions(long_positions, total_cash, df_dict)
        for equity in proportions:
            logging.info(f"Ticker: {equity['ticker']} - Proportion: {equity['proportion']:.2f}%")
    except Exception as e:
        logging.error(f"Error logging portfolio proportions: {e}")
        
        
def fetch_current_positions():
    print("Fetching current positions")
    """
    Fetch current open positions from Alpaca.
    Returns a list of dictionaries with tickers and quantities.
    """
    try:
        positions = alpaca.list_positions()
        # Extract ticker and quantity for each position
        current_positions = [{'ticker': position.symbol, 'quantity': float(position.qty)} for position in positions]
        
        # Log current positions
        logging.info("Current positions:")
        for pos in current_positions:
            logging.info(f"Ticker: {pos['ticker']}, Quantity: {pos['quantity']}")
        
        return current_positions
    except Exception as e:
        logging.error(f"Error fetching current positions: {e}")
        return []
    
# Modify `trading_strategy` to log portfolio proportions
def trading_strategy(df_dict_1m, df_dict_3m, df_dict_5m, ratio_dict_1m, ratio_dict_3m, ratio_dict_5m, trade_type):
    print("Running strategy")
    if not df_dict_1m or not df_dict_3m or not df_dict_5m:
        logging.error("Missing data for one or more timeframes. Cannot run strategy.")
        return []
        
    trades = []
    
    # Get the latest timestamp from 1m data (most granular)
    latest_date = max(df.index[-1] for df in df_dict_1m.values())
    logging.info(f"Checking positions at {latest_date}")

    # Fetch real-time current positions from Alpaca
    current_positions = fetch_current_positions()

    # Log portfolio proportions using 1m data
    log_portfolio_proportions(current_positions, fetch_account_cash(), df_dict_1m)

    # FIRST: Check existing positions for selling (using 1m data for exits)
    for pos in current_positions:
        ticker = pos['ticker']
        quantity = pos['quantity']
        if ticker in df_dict_1m:
            df = df_dict_1m[ticker]
            latest_data = df.iloc[-1]
            score = latest_data['score']
            plot_bad = latest_data['plot_bad']
            current_price = latest_data['Close']

            if abs(score - plot_bad) < 0.2:
                place_order(ticker, None, 'sell', quantity=quantity)
                logging.info(f"Closed position for {ticker} at {current_price}")
                trades.append({
                    "ticker": ticker,
                    "exitPrice": current_price,
                    "exitDate": latest_date.strftime('%Y-%m-%d %H:%M:%S')
                })

    # SECOND: Only try to open new positions if under max_positions
    if len(current_positions) >= max_positions:
        logging.info("Max concurrent positions reached, skipping new trades")
        return trades

    # Find and execute new trades using all timeframes
    best_long = find_best_long(df_dict_1m, df_dict_3m, df_dict_5m, 
                             ratio_dict_1m, ratio_dict_3m, ratio_dict_5m, 
                             latest_date)
    
    if best_long and best_long not in [pos['ticker'] for pos in current_positions]:
        # Use 1m data for entry price
        entry_price = df_dict_1m[best_long].iloc[-1]['Close']
        amount = calculate_position_size()
        
        # Log confirmation of signal across all timeframes
        logging.info(f"Signal confirmed across all timeframes for {best_long}")
        logging.info(f"1m timeframe score: {df_dict_1m[best_long].iloc[-1]['score']}")
        logging.info(f"3m timeframe score: {df_dict_3m[best_long].iloc[-1]['score']}")
        logging.info(f"5m timeframe score: {df_dict_5m[best_long].iloc[-1]['score']}")
        
        place_order(best_long, amount, 'buy')
        logging.info(f"Opened position for {best_long} at {entry_price}")
        trades.append({
            "ticker": best_long,
            "entryPrice": entry_price,
            "entryDate": latest_date.strftime('%Y-%m-%d %H:%M:%S'),
            "quantity": amount / entry_price
        })

    return trades

def clear_dataframes(data_dict, ratio_dict):
    """Clear DataFrames and free up memory"""
    if data_dict:
        for key in list(data_dict.keys()):
            del data_dict[key]
    if ratio_dict:
        for key in list(ratio_dict.keys()):
            del ratio_dict[key]
    gc.collect()

def main(trade_type):
    try:
        current_time = datetime.now()
        logging.info(f"Starting main loop at {current_time}")
        
        # Initialize dictionaries for each timeframe
        df_dict_1m = {}
        df_dict_3m = {}
        df_dict_5m = {}
        ratio_dict_1m = {}
        ratio_dict_3m = {}
        ratio_dict_5m = {}
        
        # First pass: Calculate indicators for individual tickers in all timeframes
        for ticker in tickers:
            try:
                # Fetch and process for 1m timeframe
                single_data_1m = fetch_data([ticker], '5d', '1m')
                if single_data_1m and ticker in single_data_1m:
                    df_dict_1m[ticker] = calculate_indicators(single_data_1m[ticker])
                    del single_data_1m
                
                # Fetch and process for 3m timeframe
                single_data_3m = fetch_data([ticker], '5d', '3m')
                if single_data_3m and ticker in single_data_3m:
                    df_dict_3m[ticker] = calculate_indicators(single_data_3m[ticker])
                    del single_data_3m
                
                # Fetch and process for 5m timeframe
                single_data_5m = fetch_data([ticker], '5d', '5m')
                if single_data_5m and ticker in single_data_5m:
                    df_dict_5m[ticker] = calculate_indicators(single_data_5m[ticker])
                    del single_data_5m
                
                gc.collect()
            except Exception as e:
                logging.error(f"Error processing ticker {ticker}: {e}")
                continue
        
        # Second pass: Calculate ratios for each timeframe in smaller batches
        batch_size = 10
        for timeframe_dict, ratio_dict in [(df_dict_1m, ratio_dict_1m), 
                                         (df_dict_3m, ratio_dict_3m), 
                                         (df_dict_5m, ratio_dict_5m)]:
            tickers_list = list(timeframe_dict.keys())
            for i in range(0, len(tickers_list), batch_size):
                batch_tickers = tickers_list[i:i + batch_size]
                batch_df_dict = {t: timeframe_dict[t] for t in batch_tickers}
                batch_ratios = calculate_all_ratios(batch_df_dict)
                ratio_dict.update(batch_ratios)
                gc.collect()
        
        # Modify trading_strategy call to include all timeframes
        trades = trading_strategy(df_dict_1m, df_dict_3m, df_dict_5m, 
                                ratio_dict_1m, ratio_dict_3m, ratio_dict_5m, 
                                trade_type)
        logging.info(f"Completed trades: {len(trades)}")
        print(json.dumps(trades, indent=2))
        
    except Exception as e:
        logging.error(f"Error in main loop: {e}")
    finally:
        # Clear memory for all timeframes
        for df_dict in [df_dict_1m, df_dict_3m, df_dict_5m]:
            if df_dict:
                clear_dataframes(df_dict, {})
        for ratio_dict in [ratio_dict_1m, ratio_dict_3m, ratio_dict_5m]:
            if ratio_dict:
                clear_dataframes({}, ratio_dict)
        gc.collect()

if __name__ == "__main__":
    trade_type = 'fiveMinute'
    logging.info(f"Starting application at {datetime.now()}")
    while True:
        try:
            if is_market_open():
                logging.info(f"Market is open at {datetime.now()}")
                main(trade_type)
            else:
                logging.info(f"Market is closed at {datetime.now()}")
            
            gc.collect()  # Force garbage collection after each iteration
            time.sleep(60)
        except Exception as e:
            logging.error(f"Error during execution at {datetime.now()}: {e}")
            gc.collect()  # Force garbage collection after error