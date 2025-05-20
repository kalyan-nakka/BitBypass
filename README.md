# BitBypass

This repository is the official implementation of our paper `BitBypass: A New Direction in Jailbreaking Aligned Large 
Language Models with Bitstream Camouflage`, for performing jailbreak evaluation.

## Setup

All our experiments are performed on a Ubuntu 22.04 machine. Please follow the below instructions to replicate the 
setup of our environment.

```
conda create --name bitbypass python=3.12
conda activate bitbypass
python3 -m pip install -r requirements.txt
```

To run our code, please add relevant API keys in `configs/secrets.yml` file as,

```
# Claude AI API Key
claude:   "CLAUDE_AI_API_KEY"

# Together AI API Key
together: "TOGETHER_AI_API_KEY"

# OpenAI API Key
openai:   "OPENAI_API_KEY"

# Google AI API Key
google: "GOOGLE_AI_API_KEY"
```

## Data

The experiments, namely adversarial performance, comparison with baselines and ablation study, are evaluated using 
`AdvBench-50` and `Behaviors` datasets.

The phishing content generation experiment is evaluated using `PhishyContent` dataset.

For baselines, generate jailbreaking prompts for these 2 datasets using their code.
Refer [AutoDAN](https://github.com/SheltonLiu-N/AutoDAN), [DeepInception](https://github.com/tmlr-group/DeepInception),
and [DRA](https://github.com/LLM-DRA/DRA/) repos for generating jailbreaking prompts.

**Note:** Jailbreaking prompt generation for **Base64** is present in our code. 

## Usage

To replicate our results, please follow the below steps in order.

### Step 0: Generate Data for Baselines

`IGNORE` this step, If you are `NOT` interested in Baselines.

### Step 1: Gathering Inferences from LLMs

Use the `01_gather_inferences_from_llm.py` file for gathering inferences from LLMs. The command line args for this file
are,

```
------------- | --------------------------- | -----------------------------------------
  arg         |  Description                |  Values     
------------- | --------------------------- | -----------------------------------------
--attack      |  Attack Strategy            |  bitbypass, di, autodan, deepinc, base64
              |                             |  dra, bitbypass-ab1, bitbypass-ab2,
              |                             |  bitbypass-ab3, bitbypass-ab4 
              |                             |
--dataset     |  Name of Benchmark Dataset  |  advbench_50, behaviors, phishycontent
              |                             |
--target_llm  |  Name of Target LLM         |  claude, llama, mixtral, gpt-4o, gemini
------------- | --------------------------- | -----------------------------------------
```

All inferences from each LLM will be saved in separate `JSON` files in `./results/`
folder with naming convention of `{prompting_name}-{dataset}-{target_llm}-responses.json`.

An example command for gathering inferences from `Claude` using our `BitBypass` jailbreaking prompt 
strategy for `AdvBench-50` dataset,

```
python 01_gather_inferences_from_llm.py --attack bitbypass --dataset advbench_50 --target_llm claude
```

If `INTERESTED`, gather inferences for other prompting strategies as,

```
# Direct Instruction
python 01_gather_inferences_from_llm.py --attack di --dataset DATASET --target_llm TARGET_LLM

# AutoDAN (Jailbreaking Baseline)
python 01_gather_inferences_from_llm.py --attack autodan --dataset DATASET --target_llm TARGET_LLM

# DeepInception (Jailbreaking Baseline)
python 01_gather_inferences_from_llm.py --attack deepinc --dataset DATASET --target_llm TARGET_LLM

# DRA (Jailbreaking Baseline)
python 01_gather_inferences_from_llm.py --attack dra --dataset DATASET --target_llm TARGET_LLM

# Base64 (Jailbreaking Baseline)
python 01_gather_inferences_from_llm.py --attack base64 --dataset DATASET --target_llm TARGET_LLM
```

If `INTERESTED`, gather inferences for ablations of our `BitBypass` jailbreaking prompt as,

```
# BitBypass Ablation 1
python 01_gather_inferences_from_llm.py --attack bitbypass-ab1 --dataset DATASET --target_llm TARGET_LLM

# BitBypass Ablation 2
python 01_gather_inferences_from_llm.py --attack bitbypass-ab2 --dataset DATASET --target_llm TARGET_LLM

# BitBypass Ablation 3
python 01_gather_inferences_from_llm.py --attack bitbypass-ab3 --dataset DATASET --target_llm TARGET_LLM

# BitBypass Ablation 4
python 01_gather_inferences_from_llm.py --attack bitbypass-ab4 --dataset DATASET --target_llm TARGET_LLM
```

### Step 2: Evaluate the inferences from LLMs

`IGNORE` this step, If you are `NOT` interested in following our results generation process.

Use the python file `02_eval_by_ref_and_llm_judges.py` for evaluating and compute results pertaining to Refusal-Judge
and LLM-Judge, by running the below command,

```
python 02_eval_by_ref_and_llm_judges.py
```

Use the python file `02_eval_by_hb_judge.py` for evaluating and compute results pertaining to Harm-Judge, by running
the below command, 

```
python 02_eval_by_hb_judge.py
```

### Step 3: Generate results

`IGNORE` this step, If you are `NOT` interested in following our results generation process.

Use the jupyter notebook `03_gen_results_perf_baselines_ablation.ipynb` for replicating the results of our Adversarial 
Performance, Comparison with State-of-the-Art Attacks and Ablation Study experiments.

Use the jupyter notebook `03_gen_results_bypassing_guard_models.ipynb` for replicating the results of our Bypassing 
Guard Models experiment.

Use the jupyter notebook `03_gen_results_perf_phishycontent.ipynb` for replicating the results of our Phishing Content 
Generation Performance experiment.

## Acknowledgements

Code for baseline jailbreak attacks is obtaned from [AutoDAN](https://github.com/SheltonLiu-N/AutoDAN), 
[DeepInception](https://github.com/tmlr-group/DeepInception), and [DRA](https://github.com/LLM-DRA/DRA/) repos. 

Thank you for sharing the artifacts.