# PyIRTools (Python Information Retrieval Tools)

PyIRTools is a Python library that provides a set of tools for Information Retrieval (IR) tasks. The project contains two main models: Boolean Model and Vector Space Model. The Boolean Model is a simple model that uses Boolean operators to retrieve documents. The Vector Space Model is a more complex model that uses the cosine similarity to retrieve documents. The project also contains a set of tools for text processing, such as tokenization, stop words removal, and stemming.

## Installation

This library still in development and not yet available on PyPI. You can install it by cloning the repository and running the setup script:

```bash
git clone <repository_url>
cd pyirtools
py install .
```

## Usage

This library tested using Indonesian language. Other languages may work, but the results may not be optimal and need further testing.
Here is an example of how to use the Boolean Model:

```python
from PyIRTools.model.boolean import BooleanModel

# Create a Boolean Model
boolean_model = BooleanModel("indonesian")

# Index some documents
documents = """Saya tidak masuk sekolah karena sakit.
    Saya ke sekolah berjalan kaki.
    Makanan favorit saya ayam goreng."""
boolean_model.insert_documents(documents)

# Retrieve documents using a query
query = "Sekolah OR (makan AND NOT ayam)"
boolean_model.search(query)
```

And here is an example of how to use the Vector Space Model:

```python
from PyIRTools.model.space_vector import SpaceVectorModel

# Create a Vector Space Model
vsm = SpaceVectorModel("indonesian")
query = "Stadion Lapangan Populer"
text = """Setiap akhir pekan, saya sering menonton sepak bola di stadion.
Saya suka bermain sepak bola di lapangan dekat rumah saya.
Lapangan sepak bola di kota ini sangat luas.
Sepak bola merupakan olahraga yang sangat populer di dunia.
Pemain sepak bola idola saya adalah Cristiano Ronaldo."""

# Insert documents and set query
vsm.insert_documents(text)
vsm.set_query(query)

# calculate tf-idf
df_tf = vsm.calculate_tf_idf()

# get relevant document
vsm.get_relevant_document_index(df_tf, verbose=True)

# save to excel
vsm.save_to_excel(df_tf, "cosine_similarity.xlsx")

```

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project was inspired by my assignment in the Information Retrieval course at The National Development University "Veteran" of Jakarta. This is my first project in the field of Information Retrieval, so I am still learning a lot of things. I hope any suggestions and contributions to this project are welcome.

## Future Work

- Implement more advanced IR models
- Improve the text processing tools
- Add more functionalities to the models
- Add more tests
- Insert formulas to excel files

## How to Contribute

1. Make virtual environment

    ```bash
    python -m venv env
    ```

2. Fork the repository
3. Create a new branch (`git checkout -b <branch_name>`)
4. Commit your changes (`git commit -am '<commit_message>'`)
5. Push to the branch (`git push origin <branch_name>`)
6. Create a new Pull Request
