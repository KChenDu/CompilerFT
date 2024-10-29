from os import cpu_count
from datasets import load_dataset
from setting import PROMPT_TEMPLATE
from subprocess import run
from argparse import ArgumentParser
from pathlib import Path
from json import loads, dump
from random import seed, choice


class PromptGenerator:
    @staticmethod
    def format_train_example(q: str, tests: list[str], code: str | None = None):
        prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}".format(q, '\n'.join(map(str.strip, tests)))
        if code is not None:
            prompt += f"\n>>> Code:\n```python\n{code.strip()}\n```"
        return prompt

    def __init__(self):
        num_proc = cpu_count()
        prompt_examples = load_dataset("mbpp", split="prompt[:3]", num_proc=num_proc)

        examples_str = [None, None, None]
        format_train_example = self.format_train_example

        for i in range(3):
            example_prompt = format_train_example(prompt_examples[i]['text'], prompt_examples[i]['test_list'], prompt_examples[i]['code'])
            examples_str[i] = f'- Example {i + 1}:\n{example_prompt}'

        self.examples_str = examples_str
        self.train_examples = load_dataset("mbpp", split="train", num_proc=num_proc)

    def get_prompt(self, task_id: int) -> str:
        train_examples = self.train_examples
        assert train_examples[0]['task_id'] <= task_id <= train_examples[-1]['task_id']

        example = train_examples[task_id - train_examples[0]['task_id']]
        prompt = self.format_train_example(example['text'], example['test_list'])
        prompt = PROMPT_TEMPLATE.format('\n\n'.join(self.examples_str), prompt)
        return prompt


def is_compilable(code: str) -> bool:
    with open(filename, 'w') as file:
        print(code, file=file)
    output = run(command, capture_output=True)
    return output.returncode == 0


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--compiler', default='Codon', type=str)
    parser.add_argument('--n_attempts', default=5, type=int)
    args = parser.parse_args()

    path = Path(args.path)
    assert path.is_file()

    compiler = args.compiler
    if compiler == 'Cython':
        from setting import CYTHON_COMMAND as command
        filename = Path(command[1])
    elif compiler == 'Codon':
        from setting import CODON_COMMAND as command
        filename = Path(command[-1])
    else:
        raise ValueError

    seed(42)

    train_examples = load_dataset("mbpp", split="train", num_proc=cpu_count())
    index2positives = [None] * len(train_examples)

    for i, train_example in enumerate(train_examples):
        index2positives[i] = [train_example['code']]

    offset = train_examples[0]['task_id']
    compiler_dpo_dataset_dict = {
        "prompt": [],
        "chosen": [],
        "rejected": []
    }

    prompt_generator = PromptGenerator()

    with open(path, 'r') as file:
        for line in file:
            data = loads(line)
            if len(data['generation']) < 2:
                continue
            prompt = prompt_generator.get_prompt(data['task_id'])
            generations = data['generation']
            if len(generations) < 5 or is_compilable(generations[-1]):
                for attempt in generations[:-1]:
                    compiler_dpo_dataset_dict["prompt"].append(prompt)
                    compiler_dpo_dataset_dict["chosen"].append("```python\n" + generations[-1].strip() + "\n```")
                    if attempt.lstrip().startswith('def ') \
                            or attempt.lstrip().startswith('import ') \
                            or attempt.lstrip().startswith('from') \
                            or attempt.lstrip().startswith('@') \
                            or attempt.lstrip().startswith('class ') \
                            or attempt.lstrip().startswith('#'):
                        compiler_dpo_dataset_dict["rejected"].append('```python\n' + attempt.strip() + '\n```')
                    else:
                        compiler_dpo_dataset_dict["rejected"].append(attempt.strip())
            else:
                for attempt in generations:
                    compiler_dpo_dataset_dict["prompt"].append(prompt)
                    compiler_dpo_dataset_dict["chosen"].append('```python\n' + choice(index2positives[data['task_id'] - offset]) + '\n```')
                    if attempt['generation'].lstrip().startswith('def ') \
                            or attempt.lstrip().startswith('import ') \
                            or attempt.lstrip().startswith('from') \
                            or attempt.lstrip().startswith('@') \
                            or attempt.lstrip().startswith('class ') \
                            or attempt.lstrip().startswith('#'):
                        compiler_dpo_dataset_dict["rejected"].append('```python\n' + attempt.strip() + '\n```')
                    else:
                        compiler_dpo_dataset_dict["rejected"].append(attempt.strip())

    with open('compiler_dpo_dataset_dict.json', 'w') as file:
        dump(compiler_dpo_dataset_dict, file)
