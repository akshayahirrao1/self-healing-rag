import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.critic import run_critic, CriticDecision

class TestCritic(unittest.TestCase):
    @unittest.skipIf(os.environ.get("CI") == "true", "Skipping LLM test in CI environment")
    def test_supported_answer(self):
        # A fully supported answer should be accepted
        context = "Employees receive 20 annual vacation days per year."
        question = "How many vacation days do employees get?"
        answer = "Employees receive 20 annual vacation days per year."
        
        result = run_critic(question, context, answer)
        self.assertEqual(result.decision.lower(), "accept")

    @unittest.skipIf(os.environ.get("CI") == "true", "Skipping LLM test in CI environment")
    def test_unsupported_answer(self):
        # A hallucinated answer should be rejected
        context = "Employees receive 20 annual vacation days per year."
        question = "How many vacation days do employees get?"
        answer = "Employees receive 40 annual vacation days per year."
        
        result = run_critic(question, context, answer)
        self.assertEqual(result.decision.lower(), "reject")

if __name__ == '__main__':
    unittest.main()
