import os
import argparse
import pandas as pd

from src.llm import LLM
from src.utils import (get_responses_from_llm,
                       save_data_to_json,
                       get_dataset_path,
                       get_responses_from_llm_ab1,
                       get_responses_from_llm_ab2,
                       get_responses_from_llm_di,
                       get_responses_from_llm_base64,
                       get_responses_from_llm_jb,
                       get_responses_from_llm_ab3,
                       get_responses_from_llm_ab4)


def parse_arguments():
    parser = argparse.ArgumentParser("BitBypass")

    # Attack Strategy
    parser.add_argument("--attack",
                        type=str,
                        default="bitbypass",
                        choices=["bitbypass",
                                 "bitbypass-ab1",
                                 "bitbypass-ab2",
                                 "bitbypass-ab3",
                                 "bitbypass-ab4",
                                 "di",
                                 "autodan",
                                 "deepinc",
                                 "base64",
                                 "dra"],
                        help="Attack Strategy")

    # Data
    parser.add_argument("--dataset",
                        type=str,
                        default="advbench_50",
                        choices=["advbench_50",
                                 "behaviors",
                                 "phishycontent"],
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

    ##################
    # Data File Path #
    ##################
    try:
        dataset_path = get_dataset_path(args.dataset, args.attack)
    except NameError as e:
        print(e)
        return

    ####################################
    # Load Dataset as Pandas Dataframe #
    ####################################
    dataset_df = pd.read_csv(dataset_path)
    dataset_records = dataset_df["prompt"].values.tolist()

    ###################################
    # Load the wrapper for Target LLM #
    ###################################
    target_llm = LLM(model=args.target_llm)

    ##################################################
    # Get inferences (responses) from the Target LLM #
    ##################################################

    # BitBypass
    if args.attack == "bitbypass":
        responses = get_responses_from_llm(llm=target_llm, data_records=dataset_records)

    # BitBypass Ablation Study 1
    # [BINARY_WORD] w/o Separator in User Prompt
    elif args.attack == "bitbypass-ab1":
        responses = get_responses_from_llm_ab1(llm=target_llm, data_records=dataset_records)

    # BitBypass Ablation Study 2
    # Remove Program-of-Thought in System Prompt
    elif args.attack == "bitbypass-ab2":
        responses = get_responses_from_llm_ab2(llm=target_llm, data_records=dataset_records)

    # BitBypass Ablation Study 3
    # Remove Curbed Capabilities in System Prompt
    elif args.attack == "bitbypass-ab3":
        responses = get_responses_from_llm_ab3(llm=target_llm, data_records=dataset_records)

    # BitBypass Ablation Study 4
    # Chat Interface Target-able Version
    elif args.attack == "bitbypass-ab4":
        responses = get_responses_from_llm_ab4(llm=target_llm, data_records=dataset_records)

    # Direct Instruction
    elif args.attack == "di":
        responses = get_responses_from_llm_di(llm=target_llm, data_records=dataset_records)

    # Base64
    elif args.attack == "base64":
        responses = get_responses_from_llm_base64(llm=target_llm, data_records=dataset_records)

    # AutoDAN or DeepInception or DRA
    elif args.attack in ["autodan", "deepinc", "dra"]:
        responses = get_responses_from_llm_jb(llm=target_llm,
                                              data_records=dataset_records,
                                              attack=args.attack,
                                              dataset_path=dataset_path)

    else:
        raise NotImplementedError("This attack is not implemented in this version")

    ##################################
    # Save the response in JSON file #
    ##################################
    os.makedirs("results", exist_ok=True)
    save_data_to_json(file_name=f"results/{args.attack}-{args.dataset}-{args.target_llm}-responses.json",
                      data=responses)


if __name__ == '__main__':
    main()
