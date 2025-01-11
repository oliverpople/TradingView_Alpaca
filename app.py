from flask import Flask, request, jsonify
import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
import logging

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)

# Alpaca API setup using environment variables
api = tradeapi.REST(
    os.getenv('ALPACA_API_KEY'),
    os.getenv('ALPACA_SECRET_KEY'),
    os.getenv('ALPACA_API_URL', 'https://paper-api.alpaca.markets/v2'),
    api_version='v2'
)

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

        # Verify API connection
        try:
            account = api.get_account()
            logger.info(f"Account status: {account.status}")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca API: {str(e)}")
            return jsonify({"error": "Failed to connect to trading account"}), 500

        if action == 'Buy':
            try:
                # Get the latest bar data instead of last trade
                bars = api.get_bars(ticker, '1Min', limit=1)
                if not bars:
                    logger.error(f"No price data available for {ticker}")
                    return jsonify({"error": f"No price data available for {ticker}"}), 400
                
                asset_price = float(bars[0].c)  # Use closing price
                logger.info(f"Current price for {ticker}: {asset_price}")

                # Calculate position size (use 95% of buying power to account for fees)
                buying_power = float(account.cash) * 0.95
                logger.info(f"Available buying power: {buying_power}")

                max_contracts = int(buying_power / asset_price)
                
                if max_contracts > 0:
                    # Submit the order
                    order = api.submit_order(
                        symbol=ticker,
                        qty=max_contracts,
                        side='buy',
                        type='market',
                        time_in_force='gtc'
                    )
                    logger.info(f"Buy order submitted: {order}")
                    return jsonify({
                        "message": "Trade executed successfully",
                        "order_id": order.id,
                        "quantity": max_contracts,
                        "ticker": ticker
                    }), 200
                else:
                    return jsonify({"error": "Insufficient funds"}), 400

            except Exception as e:
                logger.error(f"Error executing buy order: {str(e)}")
                return jsonify({"error": f"Failed to execute buy order: {str(e)}"}), 500

        elif action == 'Sell':
            try:
                position = api.get_position(ticker)
                qty_to_sell = int(position.qty)

                if qty_to_sell > 0:
                    order = api.submit_order(
                        symbol=ticker,
                        qty=qty_to_sell,
                        side='sell',
                        type='market',
                        time_in_force='gtc'
                    )
                    logger.info(f"Sell order submitted: {order}")
                    return jsonify({
                        "message": "Trade executed successfully",
                        "order_id": order.id,
                        "quantity": qty_to_sell,
                        "ticker": ticker
                    }), 200
                else:
                    return jsonify({"error": "No position to sell"}), 400

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