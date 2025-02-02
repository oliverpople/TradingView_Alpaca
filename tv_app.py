from flask import Flask, request, jsonify
import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
import logging
import time
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)

# Initialize Alpaca API for trading
api = tradeapi.REST(
    os.getenv('ALPACA_API_KEY'),
    os.getenv('ALPACA_SECRET_KEY'),
    'https://paper-api.alpaca.markets',
    api_version='v2'
)

def get_crypto_price(symbol):
    try:
        # Format symbol for API call (remove slash)
        api_symbol = symbol.replace('/', '')
        
        # Get latest trade from Alpaca Crypto API v2
        url = f"https://data.alpaca.markets/v2/crypto/{api_symbol}/trades/latest"
        headers = {
            "APCA-API-KEY-ID": os.getenv('ALPACA_API_KEY'),
            "APCA-API-SECRET-KEY": os.getenv('ALPACA_SECRET_KEY')
        }
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            trade_data = response.json()
            return float(trade_data['trade']['p'])
        
        raise Exception(f"Failed to get price: {response.text}")
    except Exception as e:
        logger.error(f"Error getting crypto price for {symbol}: {str(e)}")
        raise

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

        # Format display ticker (for logging)
        display_ticker = f"{ticker}/USD" if not ticker.endswith('USD') else ticker
        # Format order ticker (for API calls)
        order_symbol = display_ticker.replace('/', '')

        logger.info(f"Received signal: {signal} for ticker: {display_ticker}")

        # Verify the symbol exists by attempting to get its price
        try:
            current_price = get_crypto_price(display_ticker)
            logger.info(f"Successfully verified ticker {display_ticker} exists, current price: {current_price}")
        except Exception as e:
            logger.error(f"Invalid ticker {display_ticker}: {str(e)}")
            return jsonify({'error': f'Invalid ticker {display_ticker}'}), 400

        if signal == 'Long Open':
            # Get account information
            account = api.get_account()
            buying_power = float(account.buying_power)
            
            # Calculate quantity with 1% buffer for price movement
            quantity = int((buying_power * 0.99) / current_price)
            
            if quantity <= 0:
                return jsonify({'error': 'Insufficient buying power'}), 400

            # Place buy order
            try:
                order = api.submit_order(
                    symbol=order_symbol,
                    qty=quantity,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
                
                if wait_for_order_fill(order.id):
                    logger.info(f"Successfully opened long position for {display_ticker}")
                    return jsonify({'message': 'Long position opened successfully'}), 200
                else:
                    return jsonify({'error': 'Order failed to fill'}), 400
                
            except Exception as e:
                logger.error(f"Error placing buy order: {str(e)}")
                return jsonify({'error': str(e)}), 400

        elif signal == 'Long Close':
            # Get current position
            quantity = get_position_quantity(order_symbol)
            
            if quantity <= 0:
                return jsonify({'message': 'No position to close'}), 200

            # Place sell order
            try:
                order = api.submit_order(
                    symbol=order_symbol,
                    qty=quantity,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                
                if wait_for_order_fill(order.id):
                    logger.info(f"Successfully closed long position for {display_ticker}")
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

