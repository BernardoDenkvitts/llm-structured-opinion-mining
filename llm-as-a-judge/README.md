# LLM-as-a-Judge for Few-Shot vs Zero-Shot Comparison

`judge.ipynb` evaluates the quality of **feature-centric opinion extraction** produced by the same language model under two different prompting strategies: **few-shot** and **zero-shot**.

The comparison is performed using the **LLM-as-a-Judge** approach, where large language models act as automatic evaluators in a blind pairwise setting.

100 samples were considered for the evaluation.

Models used as judges:

- **llama-3.3-70b-versatile**
- **qwen/qwen3-32b**

results.json contains the results of the evaluation.
A - few-shot
B - zero-shot