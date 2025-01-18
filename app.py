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

def get_position_quantity(symbol):
    try:
        position = api.get_position(symbol)
        return float(position.qty)
    except Exception as e:
        logger.error(f"Error getting position for {symbol}: {str(e)}")
        return 0.0

def wait_for_order_fill(order_id, timeout=60):
    start_time = time.time()
    while time.time() - start_time < timeout:
        order = api.get_order(order_id)
        if order.status == 'filled':
            return True
        elif order.status == 'rejected' or order.status == 'canceled':
            return False
        time.sleep(1)
    return False

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'}), 400

        signal = data.get('signal')
        ticker = data.get('ticker')

        if not signal or not ticker:
            return jsonify({'error': 'Missing signal or ticker'}), 400

        # Format crypto ticker for Alpaca (add USD suffix if not present)
        if not ticker.endswith('USD'):
            ticker = f"{ticker}USD"

        logger.info(f"Received signal: {signal} for ticker: {ticker}")

        # Verify the symbol exists
        try:
            # Try to get the latest trade to verify the symbol
            api.get_latest_trade(ticker)
            logger.info(f"Successfully verified ticker {ticker} exists")
        except Exception as e:
            logger.error(f"Invalid ticker {ticker}: {str(e)}")
            return jsonify({'error': f'Invalid ticker {ticker}'}), 400

        if signal == 'Long Open':
            # Get account information
            account = api.get_account()
            buying_power = float(account.buying_power)
            
            # Get current price
            ticker_data = api.get_latest_trade(ticker)
            current_price = float(ticker_data.price)
            
            # Calculate quantity with 1% buffer for price movement
            quantity = int((buying_power * 0.99) / current_price)
            
            if quantity <= 0:
                return jsonify({'error': 'Insufficient buying power'}), 400

            # Place buy order
            try:
                order = api.submit_order(
                    symbol=ticker,
                    qty=quantity,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
                
                if wait_for_order_fill(order.id):
                    logger.info(f"Successfully opened long position for {ticker}")
                    return jsonify({'message': 'Long position opened successfully'}), 200
                else:
                    return jsonify({'error': 'Order failed to fill'}), 400
                
            except Exception as e:
                logger.error(f"Error placing buy order: {str(e)}")
                return jsonify({'error': str(e)}), 400

        elif signal == 'Long Close':
            # Get current position
            quantity = get_position_quantity(ticker)
            
            if quantity <= 0:
                return jsonify({'message': 'No position to close'}), 200

            # Place sell order
            try:
                order = api.submit_order(
                    symbol=ticker,
                    qty=quantity,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                
                if wait_for_order_fill(order.id):
                    logger.info(f"Successfully closed long position for {ticker}")
                    return jsonify({'message': 'Long position closed successfully'}), 200
                else:
                    return jsonify({'error': 'Order failed to fill'}), 400
                
            except Exception as e:
                logger.error(f"Error placing sell order: {str(e)}")
                return jsonify({'error': str(e)}), 400

        return jsonify({'error': 'Invalid signal'}), 400

    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

