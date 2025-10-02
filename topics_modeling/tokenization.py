from tqdm import tqdm
from typing import List
import argparse
import os
import time
import spacy
import polars as pl


def read_csv(csv_path: str) -> pl.DataFrame:
    return pl.read_csv(csv_path)


def process_texts(
    texts: List[str],
    nlp: spacy.language.Language,
    cores: int,
) -> List[List[str]]:
    """Remove stopwords, punctuation, digits and words with less than 3 characters

    Args:
        texts (List[str]): List of texts to process
        nlp (spacy.language.Language): NLP model to do the processing
        cores (int): Number of cores to use with nlp pipe

    Returns:
        List[List[str]]: List of processed tokens
    """
    docs = nlp.pipe(texts, n_process=cores, batch_size=1000)
    processed_tokens = []
    for doc in docs:
        processed_tokens.append(
            [
                token.lemma_.lower()
                for token in doc
                if not token.is_stop
                and not token.is_punct
                and not any(char.isdigit() for char in token.text)
                and len(token.text) > 2
            ]
        )

    return processed_tokens


def create_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv-dir",
        dest="input_csv_dir",
        help="Directory for input texts",
        required=True,
    )
    parser.add_argument(
        "--output-jsonl-dir",
        dest="output_jsonl_dir",
        help="Directory for processed output texts",
        required=True,
    )
    parser.add_argument(
        "--cores",
        type=int,
        dest="cores",
        help="Number of cores to use for processing the files in parallel",
        required=True,
        default=os.cpu_count(),
    )
    return parser.parse_args()


def get_output_filename(output_dir: str, filename: str) -> str:
    # Replace csv extension with jsonl
    return os.path.join(output_dir, filename.replace(".csv", ".jsonl"))


if __name__ == "__main__":
    args = create_config()
    os.makedirs(args.output_jsonl_dir, exist_ok=True)

    input_files = os.listdir(args.input_csv_dir)

    start_time = time.time()

    # Prepare Greek model. Keep lowercase, tokenizer and lemmatizer, download if not found

    nlp = spacy.load("el_core_news_sm", disable=["parser", "ner"])

    for file in tqdm(input_files, total=len(input_files), desc="Processing CSV files"):
        df = read_csv(os.path.join(args.input_csv_dir, file))
        texts = df["speech"].to_list()
        # Pass text through the Greek model pipe

        processed_texts = process_texts(texts, nlp, args.cores)

        # Now we assign each speaker a list of tokens in place of the original speech
        df = df.with_columns(pl.Series(processed_texts).alias("speech"))

        # Write dataframe to JSONL. Dump the tokens with orjson
        output_filename = get_output_filename(args.output_jsonl_dir, file)
        df.write_ndjson(output_filename)
