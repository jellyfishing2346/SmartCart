from smartcart.model import SmartCart
import argparse

def main():
    parser = argparse.ArgumentParser(description='SmartCart Stock Predictor')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol (e.g. AAPL)')
    parser.add_argument('--days', type=int, default=7, help='Days to predict')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    args = parser.parse_args()

    model = SmartCart(args.ticker)
    print(f"Fetching data for {args.ticker}...")
    model.fetch_data()
    model.prepare_data()
    model.build_model()
    print("Training model...")
    model.train(epochs=args.epochs)
    print(f"Predicting next {args.days} days...")
    preds = model.predict(days=args.days)
    print("Predicted prices:", [round(p, 2) for p in preds])

if __name__ == '__main__':
    main()
