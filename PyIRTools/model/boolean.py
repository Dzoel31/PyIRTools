"""
This module contains the boolean model class for information retrieval system.
"""

import re
from typing import Any, LiteralString
from pandas.core.frame import DataFrame
import inflect
import sys
import os

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(dir_path)

from utils.preprocess import Preprocess


class BooleanModel:
    def __init__(self, stopword_lang: str) -> None:
        self.preprocess = Preprocess(stopword_lang)
        self._text = ""
        self._query = ""
        self.inf = inflect.engine()

    def create_inverted_list(self) -> DataFrame:
        """
        Create inverted list from documents.

        Returns:
        DataFrame: Inverted list.
        """
        tokens = self.preprocess.preprocess_text(self.text)
        word_corpus = self.preprocess.remove_duplicate(tokens)
        list_sentences_name = [f"Id{i+1}" for i in range(len(tokens))]

        # create dictionary with key = word and value = 0
        word_check = dict.fromkeys(list_sentences_name, {})
        for keys in word_check.keys():
            word_check[keys] = dict.fromkeys(word_corpus, 0)

        # check if word exists in document give 1 else default 0
        for i in range(len(tokens)):
            for word in tokens[i]:
                word_check[f"Id{i+1}"][word] = 1

        word_check = DataFrame(word_check)
        return word_check

    def remove_punctuation(self, sentence: str) -> str:
        """
        Remove punctuation in sentence.

        Parameters:
        sentence (str): Sentence to be cleaned.

        Returns:
        str: Sentence cleaned.
        """
        return re.sub(r"[^\w\s]", "", sentence)

    def split_punctuation(self, sentence: str):
        """
        Split punctuation in sentence.

        Parameters:
        sentence (str): Sentence to be splitted.

        Returns:
        str: Sentence splitted.
        """
        # if there is punctuation, split the word
        punctuation = re.compile(r"(\w+|\W+)")
        split_sentence = punctuation.findall(sentence)
        sentence = " ".join(split_sentence)
        return sentence

    def insert_documents(self, text: str) -> None:
        """
        Insert documents.

        Parameters:
        text (str): Documents to be inserted.

        Returns:
        None
        """
        self.text = text

    def get_query(self) -> str:
        """
        Get query.

        Returns:
        str: Query.
        """
        return self._query

    def search(self, query: str) -> None:
        """
        Search query in the documents.

        Parameters:
        query (str): Query to be searched.

        Returns:
        None
        """
        self._query = query.lower()

        inverted_list = self.create_inverted_list()

        boolean_expression, result = self.boolean_model(inverted_list)
        
        print(f"Your query: {self._query}")
        print(f"Boolean Expression: {boolean_expression}")
        print(f"Result: {result}")
        if result != None:
            print(f"Result Query: {self.get_document_index(result, inverted_list)}")
        else:
            print(f"Result Query: {result}")

    def boolean_model(self, inverted_list: DataFrame) -> tuple[LiteralString, Any | None]:
        """
        Boolean model for information retrieval system.

        Parameters:
        inverted_list (DataFrame): Inverted list.

        Returns:
        tuple[str, Any]: Boolean expression and result of boolean model.
        """
        query = self._query
        regex = re.compile(r"(\band\b|\bor\b|\bnot\b)")

        # get boolean operator
        query_operator = regex.findall(query)

        queries = self.split_punctuation(query).split()

        corpus = inverted_list.index.tolist()

        tf_biner = {}

        for word in queries:
            if word in corpus:
                index = corpus.index(word)
                biner = inverted_list.iloc[index].tolist()
                biner = "".join(str(i) for i in biner)
                tf_biner[word] = biner

        boolean_operator = []

        boolean_symbol = {"and": "&", "or": "|", "not": "~"}

        for word in queries:
            if word in tf_biner.keys():
                boolean_operator.append(tf_biner[word])
            elif word in query_operator:
                boolean_operator.append(boolean_symbol[word.lower()])
            else:
                boolean_operator.append(word)

        # check if there is leading by zeros, give prefix 0o
        for i in boolean_operator:
            if i[0] == "0":
                boolean_operator[boolean_operator.index(i)] = "0o" + i

        boolean_operator = " ".join(boolean_operator)
        try:
            eval_boolean = eval(boolean_operator)
        except:
            eval_boolean = None
        return boolean_operator, eval_boolean

    def get_document_index(self, binary, inverted_list: DataFrame) -> str:
        """
        Get document index based on binary result.

        Parameters:
        binary (int): Binary result from boolean model.
        inverted_list (DataFrame): Inverted list.

        Returns:
        str: Document index based on binary result.
        """
        columms_list = inverted_list.columns.tolist()
        binary_list = [int(i) for i in str(binary)]

        index = []
        for i in range(len(binary_list)):
            if binary_list[i] == 1:
                index.append(columms_list[i])

        return self.inf.join(index)

def main():
    model = BooleanModel("indonesian")
    documents = """Saya tidak masuk sekolah karena sakit.
    Saya ke sekolah berjalan kaki.
    Makanan favorit saya ayam goreng."""

    model.insert_documents(documents)
    query = "Sekolah OR (makan AND NOT ayam)"
    model.search(query)
