from sentence_transformers import SentenceTransformer
from typing import Tuple, List
import os
import torch
import meilisearch
import numpy as np

# Needed here because of torch code causing issues with streamlit's async code
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
import streamlit as st
import time
import polars as pl


# Caches to avoid reloading data and models
@st.cache_resource
def load_model(model_path: str):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(model_path, device=device)
        if device == "cuda":
            st.info("Model moved to half precision")
        st.success(f"Loaded model : {model_path}")
        return model
    except Exception as e:
        st.error(f"Failed to load model {model_path}: {e}")
        st.stop()


@st.cache_resource
def init_meilisearch(
    client_url: str, master_key: str, index_name: str
) -> Tuple[meilisearch.Client, meilisearch.index]:
    client = meilisearch.Client(client_url, master_key)
    index = client.get_index(index_name)
    return client, index


def format_data_for_streamlit(results: List[dict]) -> pl.DataFrame:
    df = pl.DataFrame(
        results,
        schema={
            "id": pl.Utf8,
            "speaker": pl.Utf8,
            "datetime": pl.Utf8,
            "speech": pl.Utf8,
            "speech_index": pl.Int32,
            "num_words": pl.Int32,
        },
    )
    return df


# Not wrapped in main due to torch issues
# Set page settings
st.set_page_config(layout="centered", page_title="Greek Parliament speeches lookup")
st.title("Greek Parliament speeches lookup")
st.markdown(
    "Enter a query in greek and get the most similar speeches from the parliament database."
)

# Load model and database
model_path = ".models/stsb-xlm-r-greek-transfer"
st.info("Attempting to load model...")
model = load_model(model_path)
client, index = init_meilisearch(
    "http://localhost:7700", os.getenv("MEILI_MASTER_KEY"), "greek_embeddings"
)
client.index
query_text = st.text_input("Enter a query")
if query_text:
    st.write("---")
    st.subheader("Search Results:")

    with st.spinner("Searching..."):
        start_time = time.time()
        query_embeddings = model.encode(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            precision="float32",
        )
        # Quantize using sign function to +1.0 and -1.0
        query_embeddings = np.sign(query_embeddings)
        start_time = time.time()
        results = index.search(
            query_text,
            {
                "retrieveVectors": False,
                "vector": query_embeddings.tolist(),
                "hybrid": {
                    "embedder": "stsb-xlm-r-greek-transfer",
                    "semanticRatio": 0.95,
                },
            },
        )
        print(f"Searched in {time.time() - start_time} seconds")
        if results:
            # Show the results using Polars. We get the following keys: ids, documents, metadatas, distances in lists of length 10
            df: pl.DataFrame = format_data_for_streamlit(results["hits"])
            st.table(df)
        else:
            st.write("No results found. Try a different query or check the database.")
