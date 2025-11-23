import os
import argparse
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.eval import read_responses_from_json
from src.utils import save_data_to_json


MODEL_NAME_HF_URLS_MAP = {
    # Language Models
    "llama_2_lm": "meta-llama/Llama-2-7b-chat-hf",
    "llama_3_lm": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma_2_lm": "google/gemma-2-9b-it",
    "phi_3_lm": "microsoft/Phi-3.5-mini-instruct",
    "mistral_lm": "mistralai/Mistral-7B-Instruct-v0.1",

    # Guard Models
    "llama_1_gm": "meta-llama/LlamaGuard-7b",
    "llama_2_gm": "meta-llama/Meta-Llama-Guard-2-8B",
    "llama_3_gm": "meta-llama/Llama-Guard-3-8B",
    "gemma_2_gm": "google/shieldgemma-9b",
}


BENCHMARKS = [
    "advbench_50",
    "behaviors",
]


ADVERSARIAL_STRATEGIES = [
    "di",  # Direct Instructions
    "bitbypass",
    # "autodan",
    # "deepinc",
    "base64",
    # "dra",
]


class PerplexityAnalyzer:

    def __init__(self, model_name: str = "llama_2_lm"):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_HF_URLS_MAP[model_name])
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_HF_URLS_MAP[model_name])
        self.model = self.model.to('cuda')
        self.model.eval()

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def calculate_perplexity(self, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()

        return perplexity


def parse_arguments():
    parser = argparse.ArgumentParser("Perplexity Analysis")

    # Target Model
    parser.add_argument("--model",
                        type=str,
                        default="llama_2_lm",
                        choices=["llama_2_lm",
                                 "llama_3_lm",
                                 "gemma_2_lm",
                                 "phi_3_lm",
                                 "mistral_lm",
                                 "llama_1_gm",
                                 "llama_2_gm",
                                 "llama_3_gm",
                                 "gemma_2_gm",],
                        help="Name of Target Model (LM/GM)")

    return parser.parse_args()


def main():
    args = parse_arguments()

    ##################################
    # Load Perplexity Analyzer class #
    ##################################
    perplexity_analyzer = PerplexityAnalyzer(model_name=args.model)

    for benchmark in tqdm(BENCHMARKS):
        for adv_strategy in tqdm(ADVERSARIAL_STRATEGIES):

            ##################################################
            # Load prompt records for adversarial strategies #
            ##################################################

            file = f"results/{adv_strategy}-{benchmark}-claude-responses.json"
            if os.path.exists(file):
                data_records = read_responses_from_json(file)
            else:
                data_records = read_responses_from_json(f"results/di-{benchmark}-claude-responses.json")

            #############################
            # Compute Perplexity scores #
            #############################

            perplexity_results = []
            for data_record in tqdm(data_records):

                ##############################
                # Get the adversarial prompt #
                ##############################
                prompt = data_record.get(adv_strategy, None)
                if prompt is None:
                    pass

                perplexity_results.append(
                    {
                        "id": data_record.get("id"),
                        "goal": data_record.get("goal"),
                        "adv_prompt": prompt,
                        "perplexity": perplexity_analyzer.calculate_perplexity(text=prompt),
                    }
                )

            ##################################
            # Save the response in JSON file #
            ##################################
            os.makedirs("results", exist_ok=True)
            save_data_to_json(file_name=f"results/{adv_strategy}-{benchmark}-{args.model}-perplexities.json",
                              data=perplexity_results)


if __name__ == '__main__':
    main()

