import os
import argparse
import pandas as pd

from src.llm import LLM
from src.utils import (get_responses_from_llm,
                       save_responses_to_json,
                       get_dataset_path)


def parse_arguments():
    parser = argparse.ArgumentParser("Bin2Text")

    # Data
    parser.add_argument("--dataset",
                        type=str,
                        default="advbench_50",
                        choices=["advbench_50", "behaviors"],
                        help="Name of Benchmark Dataset")

    # Target LLM
    parser.add_argument("--target_llm",
                        type=str,
                        default="gpt-4o",
                        choices=["claude",
                                 "llama",
                                 "mixtral",
                                 "gpt-4o",
                                 "gemini"],
                        help="Name of Target LLM")

    return parser.parse_args()

def main():
    args = parse_arguments()

    # Data File Path
    try:
        dataset_path = get_dataset_path(args.dataset)
    except NameError as e:
        print(e)
        return

    # Load Dataset as Pandas Dataframe
    dataset_df = pd.read_csv(dataset_path)
    dataset_records = dataset_df["prompt"].values.tolist()

    # Load the wrapper for Target LLM
    target_llm = LLM(model=args.target_llm)

    # Get inferences (responses) from the Target LLM
    responses = get_responses_from_llm(llm=target_llm, data_records=dataset_records)

    # Save the inferences (responses) as JSON file
    os.makedirs("results", exist_ok=True)
    save_responses_to_json(file_name=f"results/bin2text-{args.dataset}-{args.target_llm}-responses.json",
                           responses=responses)

if __name__ == '__main__':
    main()
