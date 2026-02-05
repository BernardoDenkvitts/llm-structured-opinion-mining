# LLM Structured Opinion Extraction

A **structured opinion extraction** project using Large Language Models (LLMs) to extract opinion tuples from review/comment texts.

## Objective

Extract tuples in the format `(entity, feature, opinion, opinion_value)` from unstructured text, where:

- **Entity**: the main object about which the opinion is expressed (e.g., "iPhone 15", "restaurant")
- **Feature**: attribute or characteristic of the entity being evaluated (e.g., "battery", "customer service")
- **Opinion**: short phrase expressing the sentiment (max 5 words)
- **Opinion Value**: numerical value in the range [-1, +1] representing sentiment polarity

## Project Structure

```
├── common.py              # Helper functions for inference and processing
├── prompt.txt             # Zero-shot prompt template
├── prompt_few_shot.txt    # Few-shot prompt template with examples
├── llm-as-a-judge/        # LLM-as-a-Judge evaluation module
├── notebooks/             # Inference notebooks with different LLMs
│   ├── Qwen-Instruct.ipynb
│   ├── gemma-2-9b-it.ipynb
│   └── llama-3.1-8b.ipynb
└── results/               # Extraction results
```

## Models Used

- **Qwen2.5-7B-Instruct** & **Qwen2.5-14B-Instruct**
- **Gemma-2 9B Instruct**
- **LLaMA 3.1 8B**

## Output Example

For the input: *"Amazing tv. I love my new tv. The picture is outstanding."*

```json
{
  "opinion_tuple": [
    {
      "entity": "tv",
      "feature": "picture",
      "opinion": "outstanding",
      "opinion_value": 1.0
    }
  ]
}
```

