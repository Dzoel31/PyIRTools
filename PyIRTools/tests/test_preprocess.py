"""
Test the preprocess module.
"""

import sys
import os

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(dir_path)

from utils.preprocess import Preprocess
from pandas.core.frame import DataFrame


class TestPreprocess:
    text = "Saya sedang belajar pemrosesan bahasa alami. Pemrosesan bahasa alami adalah salah satu cabang ilmu data science."
    query = "belajar pemrosesan bahasa alami"
    preprocess = Preprocess("indonesian")

    def test_preprocess_text(self):
        """
        Test the preprocess_text method.
        """
        tokens = self.preprocess.preprocess_text(self.text)

        assert isinstance(tokens, list)
        assert isinstance(tokens[0], list)

    def test_preprocess_query(self):
        """
        Test the preprocess_query method.
        """
        query = self.preprocess.preprocess_query(self.query)

        assert isinstance(query, list)

    def test_remove_duplicate(self):
        """
        Test the remove_duplicate method.
        """
        tokens = self.preprocess.preprocess_text(self.text)
        word_corpus = self.preprocess.remove_duplicate(tokens)

        assert isinstance(word_corpus, list)

    def test_count_query(self):
        """
        Test the count_query method.
        """
        query = self.preprocess.preprocess_query(self.query)
        count_query = self.preprocess.count_query(query)

        assert isinstance(count_query, DataFrame)

    def test_count_word(self):
        """
        Test the count_word method.
        """
        tokens = self.preprocess.preprocess_text(self.text)
        count_word = self.preprocess.count_word(tokens)

        assert isinstance(count_word, DataFrame)
