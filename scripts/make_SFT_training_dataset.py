from datasets import Dataset, load_dataset
from setting import PROMPT_TEMPLATE
from os import cpu_count
from human_eval.data import write_jsonl


def read_train_examples(train_examples: Dataset, prompt_examples: Dataset) -> list[str]:
    def format_train_example(q: str, tests: list[str], code: str | None = None):
        prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}".format(q, '\n'.join(map(str.strip, tests)))
        if code is not None:
            prompt += f"\n>>> Code:\n```python\n{code.strip()}\n```"
        return prompt

    examples_str = [None, None, None]

    for i in range(3):
        example_prompt = format_train_example(prompt_examples[i]['text'], prompt_examples[i]['test_list'], prompt_examples[i]['code'])
        examples_str[i] = f'- Example {i + 1}:\n{example_prompt}'

    prompts = [None] * len(train_examples)

    for i, example in enumerate(train_examples):
        prompt = format_train_example(example['text'], example['test_list'])
        prompt = PROMPT_TEMPLATE.format('\n\n'.join(examples_str), prompt)
        completion = "```python\n" + example['code'].rstrip() + "\n```"
        yield {"prompt": prompt, "completion": completion}


if __name__ == '__main__':
    num_proc = cpu_count()
    train_examples = load_dataset("mbpp", split="train", num_proc=num_proc)
    prompt_examples = load_dataset("mbpp", split="prompt[:3]", num_proc=num_proc)
    examples = read_train_examples(train_examples, prompt_examples)

    write_jsonl("SFT_training_dataset.jsonl", examples)
