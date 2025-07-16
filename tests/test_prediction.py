import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from smartcart.model import SmartCart
import numbers

class TestSmartCart(unittest.TestCase):
    def test_predict(self):
        model = SmartCart('AAPL')
        model.fetch_data()
        model.prepare_data()
        model.build_model()
        model.train(epochs=1)  # Keep epochs low for test speed
        preds = model.predict(days=3)
        self.assertEqual(len(preds), 3)
        self.assertTrue(all(isinstance(p, numbers.Real) for p in preds))

if __name__ == '__main__':
    unittest.main()
