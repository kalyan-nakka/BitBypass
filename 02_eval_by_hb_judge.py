import os

from tqdm import tqdm

from src.utils import save_data_to_json
from src.eval import (read_responses_from_json,
                      is_response_harmful,
                      load_harmbench_jb_classifier)


LLMS = [
    "claude",
    "llama",
    "mixtral",
    "gpt-4o",
    "gemini",
]

BENCHMARKS = [
    "phishycontent",
]

JBS = [
    "bin2text",
    "di",
]


def main():
    os.makedirs("results_eval", exist_ok=True)
    harmbench_cls, harmbench_tokenizer = load_harmbench_jb_classifier()

    for benchmark in BENCHMARKS:
        for jb in JBS:
            for llm in LLMS:
                responses_json_path = f"results/{jb}-{benchmark}-{llm}-responses.json"
                raw_responses = read_responses_from_json(responses_json_path)

                response_evals = []

                for ind, raw_response in tqdm(enumerate(raw_responses)):
                    goal = raw_response.get("goal", "")
                    output = raw_response.get("output", "")

                    # Evaluation by Harmbench-Judge
                    raw_response_eval = {
                        "ind": ind,
                        "goal": goal,
                        "output": output,
                        "harmful": is_response_harmful(cls=harmbench_cls,
                                                       tokenizer=harmbench_tokenizer,
                                                       goal=goal,
                                                       response=output),
                    }
                    response_evals.append(raw_response_eval)

                # Save the evaluations to JSON for further analysis
                save_data_to_json(file_name=f"results_eval/{jb}-{benchmark}-{llm}-response-evals.json", data=response_evals)


if __name__ == '__main__':
    main()
