"""
Test the space_vector module.
"""

import sys
import os

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(dir_path)

from model.space_vector import SpaceVectorModel
from pandas.core.frame import DataFrame


class TestSpaceVector:
    model = SpaceVectorModel("indonesian")
    query = "Stadion Lapangan Populer"
    text = """Setiap akhir pekan, saya sering menonton sepak bola di stadion.
    Saya suka bermain sepak bola di lapangan dekat rumah saya.
    Lapangan sepak bola di kota ini sangat luas.
    Sepak bola merupakan olahraga yang sangat populer di dunia.
    Pemain sepak bola idola saya adalah Cristiano Ronaldo."""

    def test_space_vector(self):
        """
        Test the space_vector module.
        """
        assert isinstance(self.model, SpaceVectorModel)
        assert self.model.__class__.__name__ == "SpaceVectorModel"
        assert self.model.__module__ == "model.space_vector"

    def test_calculate_tf_idf(self):
        """
        Test the calculate_tf_idf method.
        """

        self.model.insert_documents(self.text)
        self.model.set_query(self.query)

        df_tf = self.model.calculate_tf_idf()

        assert isinstance(df_tf, DataFrame)

    def test_cosine_similarity(self):
        """
        Test the calculate_cosine_similarity method.
        """
        self.model.insert_documents(self.text)
        self.model.set_query(self.query)

        df_tf = self.model.calculate_tf_idf()
        df_similarity = self.model.calculate_cosine_similarity(df_tf)

        assert isinstance(df_similarity, dict)

    def test_save_to_excel(self):
        """
        Test the save_to_excel method.
        """
        self.model.insert_documents(self.text)
        self.model.set_query(self.query)

        df_tf = self.model.calculate_tf_idf()
        self.model.calculate_cosine_similarity(df_tf)

        self.model.save_to_excel(df_tf, "test.xlsx")

        output_path = os.path.join(self.model.OUTPUT_PATH, "test.xlsx")

        assert os.path.exists(output_path)
