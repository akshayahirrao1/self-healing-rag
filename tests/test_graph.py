import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph import route_after_critic
from src.config import Config

class TestGraphRouting(unittest.TestCase):
    def test_route_accept(self):
        state = {
            "decision": "accept",
            "retry_count": 0
        }
        self.assertEqual(route_after_critic(state), "finish")

    def test_route_reject_with_retries(self):
        state = {
            "decision": "reject",
            "retry_count": Config.MAX_RETRIES - 1
        }
        self.assertEqual(route_after_critic(state), "reformulate")

    def test_route_reject_no_retries(self):
        state = {
            "decision": "reject",
            "retry_count": Config.MAX_RETRIES
        }
        self.assertEqual(route_after_critic(state), "fallback")

if __name__ == '__main__':
    unittest.main()
