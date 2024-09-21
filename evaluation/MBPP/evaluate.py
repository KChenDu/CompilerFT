import argparse

from torch.utils.data import Dataset
from re import search, DOTALL
from setting import PROMPT_TEMPLATE
from os import cpu_count
from loguru import logger
from datasets import load_dataset
from transformers import pipeline
from pathlib import Path
from human_eval.data import write_jsonl
from human_eval.evaluation import evaluate_functional_correctness


def read_test_examples(test_examples: Dataset, prompt_examples: Dataset) -> dict:
    def format_test_example(q: str, tests: list[str], code: str | None = None):
        prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}".format(q, '\n'.join(map(str.strip, tests)))
        if code is not None:
            prompt += f"\n>>> Code:\n```python\n{code.strip()}\n```"
        return prompt

    examples_str = [None, None, None]
    for i in range(3):
        example_prompt = format_test_example(prompt_examples[i]['text'], prompt_examples[i]['test_list'], prompt_examples[i]['code'])
        examples_str[i] = f'- Example {i + 1}:\n{example_prompt}'

    for example in test_examples:
        prompt = format_test_example(example['text'], example['test_list'])
        prompt_with_shots = PROMPT_TEMPLATE.format('\n\n'.join(examples_str), prompt)
        yield prompt_with_shots


def convert_for_evaluation(generation: str) -> str:
    try:
        generation = search('```python\n.*?\n```', generation, DOTALL).group()[10:-3]
    except Exception:
        logger.warning(f"Failed to extract codeblock:\n{generation}")
    return generation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="deepseek-ai/deepseek-coder-1.3b-base", type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    args = parser.parse_args()

    num_proc = cpu_count()
    logger.info("loading MBPP dataset...")
    test_examples = load_dataset("mbpp", split="test", num_proc=num_proc)
    prompt_examples = load_dataset("mbpp", split="prompt[:3]", num_proc=num_proc)
    examples = read_test_examples(test_examples, prompt_examples)
    length = len(test_examples)
    logger.info(f"{length} problems loaded from MBPP dataset")

    model = args.model
    logger.info(f"loading model from {model}...")
    generator = pipeline('text-generation', model, device=0, batch_size=args.batch_size, torch_dtype='auto')
    logger.info(f"model loaded from {model}")

    logger.info("generating...")
    generations = generator(examples, return_full_text=False, max_new_tokens=512)

    generated_examples = [None] * length
    offset = test_examples[0]['task_id']
    for i, generation in enumerate(generations):
        generated_examples[i] = dict(task_id=offset + i, generation=convert_for_evaluation(generation[0]['generated_text']))
    logger.info("generation over")

    root = Path(__file__).parent
    logger.info("saving {} processed examples into {}...".format(length, root / "mbpp_samples.jsonl"))
    write_jsonl(root / "mbpp_samples.jsonl", generated_examples)
    logger.info("saved {} processed examples into {}".format(length, root / "mbpp_samples.jsonl"))

    result = evaluate_functional_correctness(str(root / "mbpp_samples.jsonl"), problem_file=str(root / "data" / "mbpp_test.jsonl"), is_mbpp=True)
    print(result)
