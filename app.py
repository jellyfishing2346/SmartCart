from smartcart.model import SmartCart
from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import pandas as pd
from flask import redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key in production
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Simple in-memory user store
USERS = {'testuser': {'password': 'testpass'}}

class User(UserMixin):
    def __init__(self, username):
        self.id = username
    def get_id(self):
        return self.id

@login_manager.user_loader
def load_user(user_id):
    if user_id in USERS:
        return User(user_id)
    return None

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = USERS.get(username)
        if user and user['password'] == password:
            login_user(User(username))
            flash('Logged in successfully.')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('home'))
        else:
            flash('Invalid username or password.')
    return render_template('login.html')

# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out.')
    return redirect(url_for('login'))
@app.route('/')
@login_required
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    tickers = request.form.get('ticker', '').strip()
    days = request.form.get('days', '7')
    if not tickers:
        return jsonify({'error': 'Ticker symbol(s) required.'}), 400
    try:
        days = int(days)
        if days < 1 or days > 30:
            return jsonify({'error': 'Days must be between 1 and 30.'}), 400
    except ValueError:
        return jsonify({'error': 'Days must be an integer.'}), 400

    ticker_list = [t.strip().upper() for t in tickers.split(',') if t.strip()]
    results = {}
    for ticker in ticker_list:
        try:
            model = SmartCart(ticker)
            df = model.fetch_data()
            print(f"[DEBUG] Ticker: {ticker}, DataFrame head:\n{df.head() if df is not None else df}")
            # Stricter checks for empty or invalid data
            close_col = df['Close'] if (df is not None and 'Close' in df) else None
            # If close_col is a DataFrame (multi-index), convert to Series
            if isinstance(close_col, pd.DataFrame):
                if close_col.shape[1] == 1:
                    close_col = close_col.iloc[:, 0]
            if (
                df is None or df.empty or
                close_col is None or
                not isinstance(close_col, (np.ndarray, list, pd.Series)) or
                (isinstance(close_col, pd.Series) and (close_col.empty or close_col.dropna().empty))
            ):
                # Try to give more details for debugging
                print(f"[ERROR] Data issue for ticker {ticker}: df={df}, close_col={close_col}")
                err_msg = 'No data found for this ticker.'
                if df is None:
                    err_msg += ' (Reason: DataFrame is None. Possible network or API issue.)'
                elif df.empty:
                    err_msg += ' (Reason: DataFrame is empty. Ticker may not exist or API returned no data.)'
                elif close_col is None:
                    err_msg += ' (Reason: \"Close\" column missing. Data source may have changed.)'
                elif isinstance(close_col, pd.Series) and (close_col.empty or close_col.dropna().empty):
                    err_msg += ' (Reason: \"Close\" prices are all missing or NaN.)'
                results[ticker] = {'error': err_msg + ' Please check the symbol, your network connection, or try again later.'}
                continue
            model.prepare_data()
            model.build_model()
            if not os.path.exists(model.model_path()):
                model.train(epochs=5)
            preds = model.predict(days=days)
            N = 30
            hist_prices = df['Close'].values[-N:].tolist() if 'Close' in df else []
            # Calculate metrics if possible (compare last 'days' of history to predictions)
            metrics = {}
            if hist_prices and len(hist_prices) >= days:
                actual = np.array(hist_prices[-days:])
                predicted = np.array(preds)
                rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
                mae = float(np.mean(np.abs(actual - predicted)))
                metrics = {'rmse': rmse, 'mae': mae}
            results[ticker] = {
                'predictions': preds.tolist(),
                'history': hist_prices,
                'metrics': metrics
            }
        except Exception as e:
            results[ticker] = {'error': f'Failed: {str(e)}'}
    return jsonify({'results': results})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
import sys

if __name__ == "__main__":
    port = 5000
    # Allow port override via command line argument: python3 app.py --port=XXXX
    if len(sys.argv) > 2 and sys.argv[1] == "--port":
        try:
            port = int(sys.argv[2])
        except ValueError:
            print("Invalid port specified. Using default port 5000.")
    app.run(host='0.0.0.0', port=port, debug=True)
