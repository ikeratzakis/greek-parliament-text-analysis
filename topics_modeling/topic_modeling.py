from gensim.models import LdaMulticore
from gensim.corpora import Dictionary, MmCorpus
from tqdm import tqdm
from typing import List
import time
import argparse
import orjson
import os


def prepare_corpus(jsonl_dir: str) -> List[List[str]]:
    """Load JSONL files and prepare the corpus

    Args:
        jsonl_dir (str): Directory containing the JSONL files

    Returns:
        LdaMulticore: LDA model
    """
    texts = []
    files = os.listdir(jsonl_dir)
    for filename in tqdm(files, total=len(files), desc="Loading JSONL files"):
        current_texts = []
        with open(os.path.join(jsonl_dir, filename), "rb") as file:
            for line in file:
                data = orjson.loads(line)
                current_texts.extend(data["speech"])

        texts.append(current_texts)

    return texts


def prepare_dictionary(texts: List[List[str]]) -> tuple:
    """Prepare the dictionary

    Args:
        texts (List[List[str]]): List of lists of texts

    Returns:
        dictionary: Dictionary
    """
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=10, no_above=0.5)
    dictionary.compactify()

    corpus = [
        dictionary.doc2bow(text)
        for text in tqdm(texts, total=len(texts), desc="Preparing corpus")
    ]

    return dictionary, corpus


def train_model(
    corpus: List[List[str]], dictionary: Dictionary, cores: int, num_topics: int
) -> LdaMulticore:
    """Train the LDA model

    Args:
        corpus (List[List[str]]): List of lists of texts
        dictionary (Dictionary): Dictionary

    Returns:
        LdaMulticore: LDA model
    """
    model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=10,
        workers=cores,
    )

    return model


def create_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-jsonl-dir",
        dest="input_jsonl_dir",
        help="Directory for input texts",
        required=True,
    )
    parser.add_argument(
        "--cores",
        type=int,
        dest="cores",
        help="Number of cores to use for LDAMulticore",
        required=True,
        default=os.cpu_count(),
    )
    parser.add_argument(
        "--topics",
        type=int,
        dest="topics",
        help="Number of topics to calculate",
        required=True,
    )
    return parser.parse_args()


def analyze_model(model: LdaMulticore):
    for topic_id, topic in model.show_topics(num_topics=10, num_words=10):
        print(f"Topic {topic_id}: {topic}")


if __name__ == "__main__":
    args = create_config()
    corpus = prepare_corpus(args.input_jsonl_dir)
    dictionary, corpus = prepare_dictionary(corpus)

    dictionary.save("models/greek_parliament_dictionary.dict")
    MmCorpus.serialize("models/greek_parliament_corpus.mm", corpus)

    start_time = time.time()
    model = train_model(corpus, dictionary, args.cores, args.topics)
    model.save("models/greek_parliament_model.model")
    print(f"Done in {time.time() - start_time} seconds")

    analyze_model(model)
