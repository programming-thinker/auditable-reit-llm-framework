import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.multi_agent_decision_rule import apply_multi_agent_decision_rule


class MultiAgentDecisionRuleTest(unittest.TestCase):
    def test_strong_increase(self):
        result = apply_multi_agent_decision_rule(2, 1, 1)
        self.assertEqual(result["total_score"], 4)
        self.assertAlmostEqual(result["decision_score"], 1.3333333333)
        self.assertAlmostEqual(result["allocation_score"], 1.3333333333)
        self.assertEqual(result["signal"], "increase")

    def test_strong_reduce(self):
        result = apply_multi_agent_decision_rule(-2, -1, -1)
        self.assertEqual(result["total_score"], -4)
        self.assertAlmostEqual(result["decision_score"], -1.3333333333)
        self.assertAlmostEqual(result["allocation_score"], -1.3333333333)
        self.assertEqual(result["signal"], "reduce")

    def test_hold_mixed(self):
        result = apply_multi_agent_decision_rule(1, 0, -1)
        self.assertEqual(result["total_score"], 0)
        self.assertEqual(result["decision_score"], 0)
        self.assertEqual(result["allocation_score"], 0)
        self.assertEqual(result["signal"], "hold")

    def test_high_disagreement_hold(self):
        result = apply_multi_agent_decision_rule(2, -1, -2)
        self.assertEqual(result["total_score"], -1)
        self.assertEqual(result["agent_disagreement"], 4)
        self.assertEqual(result["signal"], "hold")


if __name__ == "__main__":
    unittest.main()
