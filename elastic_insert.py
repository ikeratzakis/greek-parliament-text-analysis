import os
import argparse
import csv
from tqdm import tqdm
from typing import Dict, List, Iterable
from elasticsearch import Elasticsearch, helpers


def process_csv_file(
    filename: str, directory: str, index_name: str
) -> Iterable[Dict[str, str]]:
    # Extract the date part from the filename
    date_string = filename[
        :10
    ]  # Split the string at the first occurrence of dash or underscore
    formatted_date = date_string[:4] + "-" + date_string[5:7] + "-" + date_string[8:]

    # Parse the string into a date object
    timestamp = formatted_date
    # First column is speaker, second column is speech

    with open(os.path.join(directory, filename), "r") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            yield {
                "_index": index_name,
                "_op_type": "index",
                "_source": {
                    "speaker": row[0],
                    "speech": row[1],
                    "timestamp": timestamp,
                },
            }


def generate_insert_actions(
    files: List[str], directory: str, index_name: str
) -> Iterable[Dict[str, str]]:
    actions = []
    for file in files:
        for document in process_csv_file(file, directory, index_name):
            actions.append(document)
    return actions


def create_index(es: Elasticsearch, index_name: str, index_settings: Dict[str, str]):
    # if index exists, delete it
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    es.indices.create(index=index_name, body=index_settings)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-dir",
        dest="files_directory",
        help="Directory containing .csv files for parsing",
        required=True,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        dest="chunk_size",
        help="How many documents to process in batches to avoid RAM problems (one line = one document)",
        required=True,
        default=100,
    )
    parser.add_argument(
        "--cores",
        type=int,
        dest="cores",
        help="How many cores to use for processing/batch insertion",
        required=True,
        default=os.cpu_count(),
    )
    parser.add_argument(
        "--elastic-index",
        dest="elastic_index",
        help="Name of elasticsearch index to insert data to (will be dropped if it exists)",
        required=True,
    )
    parser.add_argument(
        "--elastic-password",
        dest="elastic_password",
        help="Elastic password for elastic superuser",
        required=True,
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Initialize Elasticsearch
    es = Elasticsearch(
        "https://localhost:9200",
        basic_auth=("elastic", args.elastic_password),
        verify_certs=False,
        ssl_show_warn=False,
        request_timeout=120,
    )

    # Basic index settings
    index_settings = {
        "mappings": {
            "properties": {
                "speaker": {
                    "type": "text",
                    "analyzer": "greek",
                    "search_analyzer": "greek",
                },
                "speech": {
                    "type": "text",
                    "analyzer": "greek",
                    "search_analyzer": "greek",
                },
                "timestamp": {
                    "type": "date",
                    "format": "yyyy-MM-dd",
                },
            },
        },
    }

    create_index(es, args.elastic_index, index_settings)

    # Insert data in batches with multiple workers
    files = os.listdir(args.files_directory)
    batch = []
    for filename in files:
        for document in process_csv_file(
            filename, args.files_directory, args.elastic_index
        ):
            batch.append(document)

            if len(batch) % args.chunk_size == 0:
                for success, info in tqdm(
                    helpers.parallel_bulk(
                        es,
                        batch,
                        chunk_size=args.chunk_size,
                        thread_count=args.cores,
                    ),
                    total=len(batch),
                    desc="Inserting docs to Elasticsearch",
                ):
                    if not success:
                        print(info)
                batch = []
    # Insert remaining documents
    if batch:
        for success, info in tqdm(
            helpers.parallel_bulk(
                es,
                batch,
                chunk_size=args.chunk_size,
                thread_count=args.cores,
            ),
            total=len(batch),
            desc="Inserting docs to Elasticsearch",
        ):
            if not success:
                print(f"Failed to insert: {info}")


if __name__ == "__main__":
    main()
