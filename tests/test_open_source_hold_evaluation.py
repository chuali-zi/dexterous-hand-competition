import unittest

from scripts.test_open_source_o6_hold import evaluate


class HoldEvaluationTest(unittest.TestCase):
    def rows(self, height=.315, release=.025, contact=True, support=False):
        return [{'t': t, 'cube': [0, 0, height if t < 8 else release],
                 'finger_contact': contact, 'support_contact': support}
                for t in [4, 5, 6.1, 8.5, 9]]

    def test_held_then_released(self):
        self.assertTrue(evaluate(self.rows())['pass'])

    def test_stationary_on_floor_is_not_a_hold(self):
        self.assertFalse(evaluate(self.rows(height=.025, contact=False))['pass'])

    def test_support_or_no_release_is_not_a_hold(self):
        self.assertFalse(evaluate(self.rows(support=True))['pass'])
        self.assertFalse(evaluate(self.rows(release=.315))['pass'])

    def test_missing_observations(self):
        self.assertFalse(evaluate([])['pass'])


if __name__ == '__main__':
    unittest.main()
