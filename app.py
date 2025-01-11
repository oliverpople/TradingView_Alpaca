from flask import Flask, request, jsonify
import alpaca_trade_api as tradeapi

app = Flask(__name__)

# Alpaca API setup
api = tradeapi.REST('PKNOCJC0383G8IQMI1X6', 'ZkA1uYK4BjPUIecrejfLS6LkS92IUOjxJTui44s8', 'https://paper-api.alpaca.markets/v2')

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        # Parse incoming JSON data
        data = request.json

        action = data.get('action')  # 'Buy' or 'Sell'
        ticker = data.get('ticker')  # Trading symbol, e.g., 'BTC/USD'

        if action == 'Buy':
            # Get the current price of the asset
            asset_price = float(api.get_last_trade(ticker).price)

            # Calculate the maximum number of contracts based on available cash
            account = api.get_account()
            buying_power = float(account.cash)
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
        print(f"Error: {str(e)}")
        return jsonify({"error": "Failed to process webhook"}), 500

if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(port=8000, debug=True)