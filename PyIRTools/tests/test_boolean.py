"""
Test the boolean module.
"""

import sys
import os

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(dir_path)

from model.boolean import BooleanModel
from pandas.core.frame import DataFrame


class TestBoolean:
    model = BooleanModel("indonesian")
    documents = """Saya tidak masuk sekolah karena sakit.
        Saya ke sekolah berjalan kaki.
        Makanan favorit saya ayam goreng."""
    query = "Sekolah OR (makan AND NOT ayam)"

    def test_boolean(self):
        """
        Test the boolean module.
        """
        assert isinstance(self.model, BooleanModel)
        assert self.model.__class__.__name__ == "BooleanModel"
        assert self.model.__module__ == "model.boolean"

    def test_inverted_list(self):
        """
        Test the create_inverted_list method.
        """
        self.model.insert_documents(self.documents)

        inverted_list = self.model.create_inverted_list()

        assert isinstance(inverted_list, DataFrame)

    def test_boolean_operator(self):
        """
        Test the boolean_model method.
        """
        self.model.insert_documents(self.documents)
        result = self.model.search(self.query)

        inverted_list = self.model.create_inverted_list()
        boolean_expression, result = self.model.boolean_model(inverted_list)

        assert isinstance(boolean_expression, str)
        assert result == 110
        assert isinstance(result, int)

    def test_boolean_get_document_index(self):
        """
        Test the get_document_index method.
        """
        self.model.insert_documents(self.documents)


        self.model.search(self.query)

        inverted_list = self.model.create_inverted_list()
        _, result = self.model.boolean_model(inverted_list)

        result = self.model.get_document_index(result, inverted_list)
        
        assert isinstance(result, str)
        assert result == "Id1 and Id2"
