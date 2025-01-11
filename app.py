from flask import Flask, request, jsonify
import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()  # Load environment variables

app = Flask(__name__)

# Alpaca API setup using environment variables
api = tradeapi.REST(
    os.getenv('ALPACA_API_KEY'),
    os.getenv('ALPACA_SECRET_KEY'),
    os.getenv('ALPACA_API_URL', 'https://paper-api.alpaca.markets/v2')
)

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        # Log the incoming request
        logger.info(f"Received webhook request: {request.json}")
        
        # Parse incoming JSON data
        data = request.json

        action = data.get('action')  # 'Buy' or 'Sell'
        ticker = data.get('ticker')  # Trading symbol, e.g., 'BTC/USD'

        logger.info(f"Processing {action} order for {ticker}")

        if action == 'Buy':
            # Get the current price of the asset
            asset_price = float(api.get_last_trade(ticker).price)
            logger.info(f"Current price for {ticker}: {asset_price}")

            # Calculate the maximum number of contracts based on available cash
            account = api.get_account()
            buying_power = float(account.cash)
            logger.info(f"Available buying power: {buying_power}")

            max_contracts = int(buying_power / asset_price)

            if max_contracts > 0:
                api.submit_order(
                    symbol=ticker,
                    qty=max_contracts,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
                print(f"Buy order submitted for {max_contracts} {ticker}")
            else:
                return jsonify({"error": "Insufficient funds"}), 400

        elif action == 'Sell':
            # Get the current position size for the ticker
            position = api.get_position(ticker)
            qty_to_sell = int(position.qty)

            if qty_to_sell > 0:
                api.submit_order(
                    symbol=ticker,
                    qty=qty_to_sell,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                print(f"Sell order submitted for {qty_to_sell} {ticker}")
            else:
                return jsonify({"error": "No position to sell"}), 400

        else:
            return jsonify({"error": "Invalid action"}), 400

        return jsonify({"message": "Trade executed successfully"}), 200

    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        return jsonify({"error": "Failed to process webhook"}), 500

if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(port=8000, debug=True)