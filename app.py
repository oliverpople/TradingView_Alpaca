from flask import Flask, request, jsonify
import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)

# Initialize Alpaca API
api = tradeapi.REST(
    os.getenv('ALPACA_API_KEY'),
    os.getenv('ALPACA_SECRET_KEY'),
    'https://paper-api.alpaca.markets',
    api_version='v2'
)

def get_latest_price(symbol):
    """Get latest price for a symbol using appropriate method"""
    try:
        # Try to get barset (for stocks)
        barset = api.get_barset(symbol, 'minute', limit=1)
        if barset and symbol in barset:
            return float(barset[symbol][0].c)
        
        logger.info("Could not get stock bars, trying crypto bars...")
        
        # If barset doesn't work, try crypto bars
        end = datetime.now()
        start = end - timedelta(minutes=1)
        bars = api.get_crypto_bars(symbol, 'minute', start=start.isoformat(), end=end.isoformat()).df
        if not bars.empty:
            return float(bars['close'].iloc[-1])
            
        logger.error(f"No price data available for {symbol}")
        return None
        
    except Exception as e:
        logger.error(f"Error getting price for {symbol}: {e}")
        return None

@app.route('/')
def home():
    return "TradingView Webhook Server is running!"

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        logger.info(f"Received webhook request: {request.json}")
        
        if not request.json:
            logger.error("No JSON data received")
            return jsonify({"error": "No JSON data received"}), 400

        data = request.json
        action = data.get('action')
        ticker = data.get('ticker')

        if not action or not ticker:
            logger.error(f"Missing required fields. Received: {data}")
            return jsonify({"error": "Missing action or ticker"}), 400

        logger.info(f"Processing {action} order for {ticker}")

        if action == 'Buy':
            try:
                # Get account info
                account = api.get_account()
                buying_power = float(account.cash)
                logger.info(f"Available buying power: ${buying_power}")

                if buying_power > 0:
                    # Get current price
                    current_price = get_latest_price(ticker)
                    if not current_price:
                        return jsonify({"error": f"Could not get price for {ticker}"}), 400
                    
                    logger.info(f"Current price for {ticker}: ${current_price}")

                    # Calculate maximum shares (use 95% of buying power)
                    max_shares = int(buying_power * 0.95 / current_price)
                    logger.info(f"Attempting to buy {max_shares} shares of {ticker}")

                    if max_shares > 0:
                        # Submit market order
                        order = api.submit_order(
                            symbol=ticker,
                            qty=max_shares,
                            side='buy',
                            type='market',
                            time_in_force='gtc'
                        )
                        logger.info(f"Buy order submitted: {order}")
                        return jsonify({
                            "message": "Buy order executed successfully",
                            "order_id": order.id,
                            "shares": max_shares,
                            "estimated_cost": max_shares * current_price,
                            "ticker": ticker
                        }), 200
                    else:
                        return jsonify({"error": "Insufficient funds for minimum order"}), 400
                else:
                    return jsonify({"error": "Insufficient funds"}), 400

            except Exception as e:
                logger.error(f"Error executing buy order: {str(e)}")
                return jsonify({"error": f"Failed to execute buy order: {str(e)}"}), 500

        elif action == 'Sell':
            try:
                # Get current position
                position = api.get_position(ticker)
                
                # Submit order to sell entire position
                order = api.submit_order(
                    symbol=ticker,
                    qty=position.qty,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                logger.info(f"Sell order submitted: {order}")
                return jsonify({
                    "message": "Sell order executed successfully",
                    "order_id": order.id,
                    "quantity": position.qty,
                    "ticker": ticker
                }), 200

            except Exception as e:
                logger.error(f"Error executing sell order: {str(e)}")
                return jsonify({"error": f"Failed to execute sell order: {str(e)}"}), 500
        else:
            return jsonify({"error": "Invalid action"}), 400

    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        return jsonify({"error": f"Failed to process webhook: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(port=8000, debug=True)