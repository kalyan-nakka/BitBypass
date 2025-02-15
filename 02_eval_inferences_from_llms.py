import os
from src.utils import save_data_to_json
from src.eval import (response_refusal_rate,
                      jailbreak_success_rate,
                      load_harmbench_jb_classifier)


LLMS = [
    "claude",
    "llama",
    "mixtral",
    "gpt-4o",
    "gemini",
]

BENCHMARKS = [
    "advbench_50",
    "behaviors",
]

JBS = [
    "bin2text",
    "di",
    "autodan",
    "deepinc",
]


def main():
    harmbench_cls = load_harmbench_jb_classifier()
    for benchmark in BENCHMARKS:
        evals = {}
        for jb in JBS:
            each_llm_eval = {}
            for llm in LLMS:
                each_llm_eval[llm] = {
                    "rrr": response_refusal_rate(target_llm=llm, dataset=benchmark, jb=jb),
                    "jsr": jailbreak_success_rate(target_llm=llm, dataset=benchmark, jb=jb, classifier=harmbench_cls),
                }
            evals[jb] = each_llm_eval

        os.makedirs("results", exist_ok=True)
        save_data_to_json(file_name=f"results/{benchmark}-llms-performance-comparison.json", data=evals)


if __name__ == '__main__':
    main()
