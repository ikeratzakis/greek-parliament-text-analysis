from tqdm import tqdm
import meilisearch
import polars as pl
import argparse
import os
import time


def generate_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings-dir",
        dest="embeddings_dir",
        help="Directory for input embeddings",
        required=True,
    )
    parser.add_argument(
        "--master-key",
        dest="master_key",
        help="Master key for meilisearch",
        required=True,
    )
    return parser.parse_args()


def create_index(client: meilisearch.Client):
    try:
        index = client.get_index("greek_embeddings")
        client.wait_for_task(task.task_uid)
        print(f"Index {index.uid} already exists")
    except Exception as e:
        task = client.create_index("greek_embeddings", {"primaryKey": "id"})
        client.wait_for_task(task.task_uid)
        index = client.get_index("greek_embeddings")
        print(f"Index {index.uid} created")

    print("Configuring index settings")
    task = index.update_settings(
        {
            "embedders": {
                "stsb-xlm-r-greek-transfer": {
                    "source": "userProvided",
                    "dimensions": 768,
                    "binaryQuantized": True,
                }
            },
            "filterableAttributes": ["speaker", "num_words"],
            "sortableAttributes": ["datetime"],
            "searchableAttributes": ["speech"],
        }
    )
    client.wait_for_task(task.task_uid)
    print("Index settings configured")
    return index


def insert_to_meilisearch(embeddings_dir: str, index: meilisearch.index):
    parquet_files = [
        os.path.join(embeddings_dir, f)
        for f in os.listdir(embeddings_dir)
        if f.endswith(".parquet")
    ]
    cur_id = 0
    for i, file in enumerate(
        tqdm(parquet_files, total=len(parquet_files), desc="Inserting to Meilisearch")
    ):
        # We will use a different id (int) because meilisearch does not play well with specific string characters
        df = (
            pl.read_parquet(
                file,
            )
            .rename({"embedding": "_vectors"})
            .select(
                [
                    "speaker",
                    "datetime",
                    "speech",
                    "speech_index",
                    "_vectors",
                    "num_words",
                ]
            )
        )
        rows = df.rows(named=True)
        # Insert to meilisearch in batches of 1000
        for batch in [rows[i : i + 1000] for i in range(0, len(rows), 1000)]:
            prepared_batch = []
            for document in batch:
                cur_id += 1
                prepared_batch.append(
                    {
                        "id": cur_id,
                        "speaker": document["speaker"],
                        "datetime": document["datetime"],
                        "speech": document["speech"],
                        "speech_index": document["speech_index"],
                        "_vectors": {"stsb-xlm-r-greek-transfer": document["_vectors"]},
                        "num_words": document["num_words"],
                    }
                )

            index.add_documents(prepared_batch)


if __name__ == "__main__":
    args = generate_config()
    client = meilisearch.Client("http://localhost:7700", args.master_key)
    index = create_index(client)
    start_time = time.time()
    insert_to_meilisearch(args.embeddings_dir, index)
    print(f"Done in {time.time() - start_time} seconds")
