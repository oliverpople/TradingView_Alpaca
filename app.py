from flask import Flask, request, jsonify
import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
import logging
import time

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

# Fixed amount to trade (in dollars)
TRADE_AMOUNT = 100.00

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

        # Ensure proper crypto ticker format (BTC/USD -> BTCUSD)
        if '/' in ticker:
            ticker = ticker.replace('/', '')
        
        logger.info(f"Processing {action} order for {ticker}")

        if action == 'Buy':
            try:
                # Get account info
                account = api.get_account()
                buying_power = float(account.cash)
                logger.info(f"Available buying power: ${buying_power}")

                if buying_power >= TRADE_AMOUNT:
                    # Submit market order using dollar amount (supports fractional shares)
                    order = api.submit_order(
                        symbol=ticker,
                        notional=TRADE_AMOUNT,  # Use fixed dollar amount for fractional shares
                        side='buy',
                        type='market',
                        time_in_force='gtc'
                    )
                    logger.info(f"Buy order submitted: {order}")
                    
                    # Wait briefly for order to process
                    time.sleep(2)
                    
                    # Check order status
                    filled_order = api.get_order(order.id)
                    logger.info(f"Order status: {filled_order.status}, filled quantity: {filled_order.filled_qty}")
                    
                    if filled_order.status == 'filled':
                        return jsonify({
                            "message": "Buy order executed successfully",
                            "order_id": order.id,
                            "amount": TRADE_AMOUNT,
                            "filled_qty": filled_order.filled_qty,
                            "filled_price": filled_order.filled_avg_price,
                            "ticker": ticker
                        }), 200
                    else:
                        logger.error(f"Order not filled. Status: {filled_order.status}")
                        return jsonify({"error": f"Order not filled. Status: {filled_order.status}"}), 500
                else:
                    return jsonify({"error": "Insufficient funds"}), 400

            except Exception as e:
                logger.error(f"Error executing buy order: {str(e)}")
                return jsonify({"error": f"Failed to execute buy order: {str(e)}"}), 500

        elif action == 'Sell':
            try:
                try:
                    # Get current position
                    position = api.get_position(ticker)
                    logger.info(f"Current position: {position.qty} {ticker} at market value ${position.market_value}")
                except Exception as pos_e:
                    logger.error(f"Error getting position: {str(pos_e)}")
                    return jsonify({"error": "No position found to sell"}), 404
                
                if float(position.qty) <= 0:
                    return jsonify({"error": "No position to sell"}), 400
                
                # Submit order to sell entire position using notional amount for better fractional handling
                order = api.submit_order(
                    symbol=ticker,
                    notional=float(position.market_value),  # Sell entire position value
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                logger.info(f"Sell order submitted: {order}")
                
                # Wait briefly for order to process
                time.sleep(2)
                
                # Check order status
                filled_order = api.get_order(order.id)
                logger.info(f"Order status: {filled_order.status}, filled quantity: {filled_order.filled_qty}")
                
                if filled_order.status == 'filled':
                    return jsonify({
                        "message": "Sell order executed successfully",
                        "order_id": order.id,
                        "quantity": filled_order.filled_qty,
                        "filled_price": filled_order.filled_avg_price,
                        "ticker": ticker
                    }), 200
                else:
                    logger.error(f"Order not filled. Status: {filled_order.status}")
                    return jsonify({"error": f"Order not filled. Status: {filled_order.status}"}), 500

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