import unittest

from src.pronunciation.scoring import assess_pronunciation


class PronunciationScoringTests(unittest.TestCase):
    def test_exact_words_are_green_and_score_full_marks(self):
        result = assess_pronunciation("Hello, world!", "hello world")

        self.assertEqual(result.accuracy, 100)
        self.assertEqual([word.color for word in result.words], ["green", "green"])

    def test_close_substitution_is_yellow_and_lowers_score(self):
        result = assess_pronunciation("cat", "cap")

        self.assertEqual(result.accuracy, 0)
        self.assertEqual(result.words[0].color, "yellow")

    def test_unrecognized_word_is_red(self):
        result = assess_pronunciation("good morning", "good")

        self.assertEqual(result.accuracy, 50)
        self.assertEqual([word.color for word in result.words], ["green", "red"])

    def test_extra_words_reduce_score(self):
        result = assess_pronunciation("hello", "hello there")

        self.assertEqual(result.accuracy, 0)
        self.assertEqual(result.words[0].color, "green")

    def test_reference_without_words_is_rejected(self):
        with self.assertRaises(ValueError):
            assess_pronunciation("...", "")


if __name__ == "__main__":
    unittest.main()
