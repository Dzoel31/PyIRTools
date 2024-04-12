from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd

import sys
import os

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(dir_path)
from pandas.core.frame import DataFrame


class Preprocess:
    def __init__(self, stopword_lang: str) -> None:
        self.stopwords_list = stopwords.words(stopword_lang)
        self.factory = StemmerFactory()
        self.stemmer = self.factory.create_stemmer()

    def preprocess_text(self, text: str) -> list[list[str]]:
        """
        Preprocess text by tokenizing, removing stopwords, and stemming.

        Parameters:
        text (str): text to be preprocessed.

        Returns:
        list[list[str]]: list of list of words in each sentence.
        """
        list_sentences = text.lower().split(".")

        # tokenize each sentence
        tokens = [word_tokenize(sentence) for sentence in list_sentences]

        # remove stopwords
        for i in range(len(tokens)):
            tokens[i] = [word for word in tokens[i] if word not in self.stopwords_list]

        # stemming
        for i in range(len(tokens)):
            tokens[i] = [self.stemmer.stem(word) for word in tokens[i]]

        # remove empty string in list
        for i in range(len(tokens)):
            tokens[i] = [word for word in tokens[i] if word]

        # remove empty list
        tokens = [sentence for sentence in tokens if sentence]
        return tokens

    def preprocess_query(self, query: str) -> list[str]:
        """
        Preprocess query by tokenizing, removing stopwords, and stemming.

        Parameters:
        query (str): query to be preprocessed.

        Returns:
        list[str]: list of words in query.
        """
        # tokenize each sentence
        list_query = word_tokenize(query.lower())

        # remove stopwords
        list_query = [word for word in list_query if word not in self.stopwords_list]

        # stemming
        list_query = [self.stemmer.stem(word) for word in list_query]

        # remove empty list
        list_query = [sentence for sentence in list_query if sentence]

        return list_query

    def remove_duplicate(self, tokens: list[list[str]]) -> list[str]:
        """
        Remove duplicate words in list of list of words.

        Parameters:
        tokens (list[list[str]]): list of list of words.

        Returns:
        list[str]: list of words without duplicate.
        """
        word_corpus = []
        for sentence in tokens:
            for word in sentence:
                if word not in word_corpus:
                    word_corpus.append(word)
        return word_corpus

    def count_query(self, word_list: list[str]) -> DataFrame:
        """
        Count word in query.

        Parameters:
        word_list (list[str]): list of words in query.

        Returns:
        DataFrame: DataFrame contain word count in query.
        """
        # remove duplicate
        word_count = dict.fromkeys(word_list, 0)
        for word in word_list:
            word_count[word] = word_list.count(word)

        return pd.DataFrame(word_count, index=["Query"]).T

    def count_word(self, word_list: list[list[str]]) -> DataFrame:
        """
        Count word in each sentence.

        Parameters:
        word_list (list[list[str]]): list of list of words.

        Returns:
        DataFrame: DataFrame contain word count in each sentence.
        """
        length = len(word_list)

        # remove duplicate
        word_corpus = self.remove_duplicate(word_list)
        sentences_code = [f"D{i+1}" for i in range(length)]
        word_count = dict.fromkeys(sentences_code, {})
        for keys in word_count:
            word_count[keys] = dict.fromkeys(word_corpus, 0)
        for i in range(length):
            for word in word_list[i]:
                word_count[sentences_code[i]][word] += 1

        # convert to dataframe
        df = pd.DataFrame(word_count).fillna(0)
        return df


# Example
def main():
    text = "Saya sedang belajar pemrosesan bahasa alami. Pemrosesan bahasa alami adalah salah satu cabang ilmu data science."
    query = "belajar pemrosesan bahasa alami"
    preprocess = Preprocess("indonesian")
    tokens = preprocess.preprocess_text(text)
    query = preprocess.preprocess_query(query)
    word_corpus = preprocess.remove_duplicate(tokens)
    count_query = preprocess.count_query(query)
    count_word = preprocess.count_word(tokens)

    print(tokens)
    print(query)
    print(word_corpus)
    print(count_query)
    print(count_word)
