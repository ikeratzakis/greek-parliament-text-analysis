from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import List
import torch
import time
import os
import argparse
import polars as pl


def generate_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv-dir",
        dest="input_csv_dir",
        help="Directory for input speeches",
        required=True,
    )
    parser.add_argument(
        "--output-embeddings-dir",
        dest="output_embeddings_dir",
        help="Directory for processed output texts",
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Batch size of embeddings to store per file",
        required=False,
    )
    parser.add_argument(
        "--model-batch-size",
        type=int,
        default=256,
        required=False,
        help="Model batch size for parallel inference",
    )
    return parser.parse_args()


def generate_model_config():
    model_name = "lighteternal/stsb-xlm-r-greek-transfer"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model_dir = ".models/stsb-xlm-r-greek-transfer"
    if not os.listdir(model_dir):
        model = SentenceTransformer(model_name, device=device)
        model.save(".models/stsb-xlm-r-greek-transfer")
    else:
        try:
            # Try to load model from directory
            model = SentenceTransformer(model_dir, device=device)
        except Exception as e:
            print(f"Failed to load model from directory: {e}. Downloading model...")
            model = SentenceTransformer(model_name, device=device)
            model.save(model_dir)

    return model


def embed_batch(
    embeddings_data: List[dict],
    model: SentenceTransformer,
    model_batch_size: int,
    output_dir: str,
    batch_index: int,
) -> int:
    # Send texts for inference
    texts_for_inference = [speech["speech"] for speech in embeddings_data]
    embeddings = model.encode(
        texts_for_inference,
        batch_size=model_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    df = pl.DataFrame(
        embeddings_data,
        schema={
            "id": pl.Utf8,
            "speaker": pl.Utf8,
            "datetime": pl.Utf8,
            "speech": pl.Utf8,
            "speech_index": pl.Int32,
        },
    )
    # Add the embeddings column and add the number of words as a column
    df = df.with_columns(
        pl.Series(embeddings, dtype=pl.Array(pl.Float32, 768)).alias("embedding"),
        pl.col("speech").str.split(" ").list.len().alias("num_words"),
    )

    output_filename = os.path.join(output_dir, f"speeches_{batch_index}.parquet")
    df.write_parquet(output_filename, compression="zstd", compression_level=6)
    batch_index += 1
    return batch_index


def generate_embeddings(
    csv_dir: str, output_dir: str, batch_size: int, model_batch_size: int
):
    # Read CSV files and generate embeddings
    model = generate_model_config()
    files = [
        os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith(".csv")
    ]
    embeddings_data = []
    batch_index = 0
    for file in tqdm(
        files, total=len(files), desc="Collecting speech data for inference"
    ):
        # The filename is the datetime, in string format
        datetime = os.path.basename(file).split(".")[0]
        df = pl.read_csv(file)
        for index, row in enumerate(df.iter_rows()):
            # Concat datetime with index to create a unique id of the speech
            speech_id = datetime + "_" + str(index)
            speech_text = row[1]
            embeddings_data.append(
                {
                    "id": speech_id,
                    "speaker": row[0],
                    "datetime": datetime[:10],
                    "speech": speech_text,
                    "speech_index": index,
                }
            )

        if len(embeddings_data) >= batch_size:
            batch_index = embed_batch(
                embeddings_data, model, model_batch_size, output_dir, batch_index
            )
            embeddings_data = []
    # Final batch
    if embeddings_data:
        batch_index = embed_batch(
            embeddings_data, model, model_batch_size, output_dir, batch_index
        )
        embeddings_data = []


if __name__ == "__main__":
    args = generate_config()
    os.makedirs(args.output_embeddings_dir, exist_ok=True)
    start_time = time.time()
    generate_embeddings(
        args.input_csv_dir,
        args.output_embeddings_dir,
        args.batch_size,
        args.model_batch_size,
    )
    print(f"Generated embeddings in {time.time() - start_time} seconds")
