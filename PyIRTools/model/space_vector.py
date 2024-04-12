"""
This module contains the space vector model class for information retrieval system.
"""

import os
import math
from typing import Any
import numpy as np

import sys
import os

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(dir_path)

from utils.preprocess import Preprocess
from pandas.core.frame import DataFrame


class SpaceVectorModel:
    """
    Space Vector Model class for Information Retrieval System.

    Attributes:
    stopword_lang (str): Stopword language.
    """
    OUTPUT_PATH = "./out"

    def __init__(self, stopword_lang: str) -> None:
        self.preprocess = Preprocess(stopword_lang)
        self._df_text: str
        self._df_query: DataFrame

    def insert_documents(self, text: str) -> None:
        """
        Insert documents to be processed.

        Parameters:
        text (str): Text documents.

        Returns:
        None
        """
        list_text = self.preprocess.preprocess_text(text)
        self.df_text = self.preprocess.count_word(list_text)

    def set_query(self, query: str) -> None:
        """
        Set query to be processed.

        Parameters:
        query (str): Query to be processed.

        Returns:
        None
        """
        list_query = self.preprocess.preprocess_query(query)
        self.df_query = self.preprocess.count_query(list_query)

    def calculate_tf_idf(self) -> DataFrame:
        """
        Calculate TF-IDF from documents and query.

        Returns:
        DataFrame: DataFrame contain TF-IDF.
        """

        df_tf = self.df_text
        df_tf.sort_index(inplace=True)

        # preprocess df
        df_tf.insert(0, "Query", self.df_query["Query"])
        df_tf.index.name = "Term"
        df_tf = df_tf.fillna(0)  # fill NaN with 0

        # change float to int
        df_tf = df_tf.astype(int)

        # calculate tf
        term_list = df_tf.columns.to_list()  # get column name

        df_tf["DF"] = df_tf.sum(axis=1)

        df_tf["IDF"] = df_tf["DF"].apply(
            lambda x: round(math.log10(len(term_list) / x), 5)
        )

        for term in term_list:
            df_tf[f"TF-IDF {term}"] = df_tf[term] * df_tf["IDF"]

        for term in term_list:
            df_tf[f"Norm {term}"] = df_tf[f"TF-IDF {term}"] ** 2

        df_tf.loc["Total"] = df_tf.sum()
        df_tf.loc["Square Root"] = df_tf.loc["Total"] ** 0.5

        return df_tf

    def calculate_cosine_similarity(self, df: DataFrame) -> dict[Any, Any]:
        """
        Calculate cosine similarity between query and documents.

        Parameters:
        df (DataFrame): DataFrame contain TF-IDF.

        Returns:
        dict[Any, Any]: Cosine similarity between query and documents.
        """
        df_similarity = df.copy(deep=False)

        # Get query index with value 1
        query_index = df_similarity.index[df_similarity["Query"] == 1].tolist()

        # calculate cosine similarity
        # get Square Root of query
        query_norm = df_similarity.loc["Square Root", "Norm Query"]

        query_norm = np.round(query_norm, 6)  # type: ignore

        # get index of Norm Query column
        index_query = df_similarity.columns.get_loc("Norm Query")

        # get column after Norm Query
        column_after_query = df_similarity.columns[index_query + 1 :].to_list()  # type: ignore

        # get value from each column after Norm Query when in query index
        # and divide it with query_norm + Square Root of each column
        result_cosine = {}
        for column in column_after_query:
            sum_value = 0
            for index in query_index:
                index_value = df_similarity.loc[index, column]
                sum_value += np.round(index_value, 6)  # type: ignore

            cosine = sum_value / (query_norm * df_similarity.loc["Square Root", column])
            result_cosine[column] = round(cosine, 6)
        
        return result_cosine

    def get_relevant_document_index(self, df: DataFrame, verbose: bool = False) -> None:
        """
        Get relevant document index based on cosine similarity.

        Parameters:
        df (DataFrame): DataFrame contain TF-IDF.
        verbose (bool): If True, print all relevant documents in Dataframe format. If False, print one most relevant document.

        Returns:
        None
        """

        cosine = self.calculate_cosine_similarity(df)

        # sort cosine similarity
        cosine = dict(sorted(cosine.items(), key=lambda item: item[1], reverse=True))

        if verbose: 
            print("Relevant Document:")
            result = DataFrame(cosine.items(), columns=["Document", "Cosine Similarity"])
            print(result)
        else:
            print(f"Most Relevant Document: {list(cosine.keys())[0]}")
        


    def save_to_excel(self, df: DataFrame, filename: str) -> None:
        """
        Save DataFrame to Excel file.

        Parameters:
        df (DataFrame): DataFrame to be saved.
        filename (str): Excel filename.

        Returns:
        None

        Note:
        Dataframe contain all cell with float value. You must clean it manually for Total and Square Root row.
        """

        if not os.path.exists(self.OUTPUT_PATH):
            os.makedirs(self.OUTPUT_PATH)

        df.to_excel(os.path.join(self.OUTPUT_PATH, filename))


def main():
    svm = SpaceVectorModel("indonesian")
    query = "Stadion Lapangan Populer"
    text = """Setiap akhir pekan, saya sering menonton sepak bola di stadion.
    Saya suka bermain sepak bola di lapangan dekat rumah saya.
    Lapangan sepak bola di kota ini sangat luas.
    Sepak bola merupakan olahraga yang sangat populer di dunia.
    Pemain sepak bola idola saya adalah Cristiano Ronaldo."""

    svm.insert_documents(text)
    svm.set_query(query)

    df_tf = svm.calculate_tf_idf()

    # get relevant document
    svm.get_relevant_document_index(df_tf, verbose=True)

    svm.save_to_excel(df_tf, "cosine_similarity.xlsx")
