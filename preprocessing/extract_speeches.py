from multiprocessing import Pool
import re
import argparse
import os
import csv
import time


def create_regex_pattern() -> str:
    # Match the speaker's name (with optional parentheses) followed by a colon
    return r"([A-ZΑ-Ω]+(?: \([^)]+\))?):\s*(.*)"


# Extract speakers and speeches
def extract_speakers_and_speeches(
    text: str, pattern: re.Pattern
) -> list[tuple[str, str]]:
    """
    Extracts speaker names and their corresponding speeches from a given text.

    Args:
        text: The text to extract speakers and speeches from.
        pattern: The compiled regular expression pattern to use for extraction.
    Returns:
        A list of tuples where each tuple contains a speaker name (str) and their speech (str).
    """
    matches = re.findall(pattern, text, re.MULTILINE)

    # Create a list of (speaker, speech) pairs. Remove stopwords from the speech
    speakers_and_speeches = [
        (speaker.strip(), speech.strip()) for speaker, speech in matches
    ]

    return speakers_and_speeches


def extract_date_from_filename(filename: str) -> str:
    # Extract the date part from the filename
    date_string = filename[:10]
    # Split the string at the first occurrence of dash or underscore
    formatted_date = date_string[:4] + "-" + date_string[5:7] + "-" + date_string[8:]
    return formatted_date


def get_output_filename(out_dir: str, filename: str) -> str:
    date_string = extract_date_from_filename(filename)
    return os.path.join(out_dir, f"{date_string+filename[11:]}".replace(".txt", ".csv"))


def write_to_file(
    speakers_and_speeches: list[tuple[str, str]], filename: str, out_dir: str
) -> None:
    with open(get_output_filename(out_dir, filename), "w") as f:
        csv_writer = csv.writer(f)
        # Write header
        csv_writer.writerow(["speaker", "speech"])
        for speaker, speech in speakers_and_speeches:
            csv_writer.writerow([speaker, speech])


def process_file(
    filename: str, input_texts_dir: str, out_dir: str, pattern: str
) -> None:
    with open(os.path.join(input_texts_dir, filename), "r") as file:
        text = file.read()
        speakers_and_speeches = extract_speakers_and_speeches(text, pattern)
        write_to_file(speakers_and_speeches, filename, out_dir)


# Configure the parser
def create_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-texts-dir",
        dest="input_texts_dir",
        help="Directory for input texts",
        required=True,
    )
    parser.add_argument(
        "--output-csv-dir",
        dest="output_csv_dir",
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


if __name__ == "__main__":
    args = create_config()
    os.makedirs(args.output_csv_dir, exist_ok=True)

    input_files = os.listdir(args.input_texts_dir)

    start_time = time.time()

    with Pool() as pool:
        pool.starmap(
            process_file,
            [
                (
                    filename,
                    args.input_texts_dir,
                    args.output_csv_dir,
                    create_regex_pattern(),
                )
                for filename in input_files
            ],
        )

    print(f"Processed {len(input_files)} files in {time.time() - start_time} seconds")
