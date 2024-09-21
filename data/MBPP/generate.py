import argparse

from setting import PROMPT_TEMPLATE, generate_kwargs
from re import search, DOTALL
from os import cpu_count
from datasets import load_dataset, Dataset
from transformers import pipeline, set_seed
from loguru import logger
from human_eval.data import write_jsonl
from pathlib import Path
from subprocess import run


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
        prompts[i] = PROMPT_TEMPLATE.format('\n\n'.join(examples_str), prompt)

    return prompts


def convert_for_evaluation(generation: str) -> str:
    try:
        generation = search('```python\n.*?\n```', generation, DOTALL).group()[10:-3]
    except Exception:
        logger.warning(f"Failed to extract codeblock:\n{generation}")
    return generation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='deepseek-ai/deepseek-coder-1.3b-base', type=str)
    parser.add_argument('--num_samples_per_task', default=20, type=int)
    parser.add_argument('--num_attempts', default=5, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--compiler', choices=('Cython', 'Codon'), default='Codon', type=str)
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    compiler = args.compiler

    if compiler == 'Cython':
        from setting import CYTHON_COMMAND as command
    elif compiler == 'Codon':
        from setting import CODON_COMMAND as command
    else:
        raise ValueError

    num_proc = cpu_count()

    if args.demo:
        prompts = read_train_examples(load_dataset("mbpp", split="train[:32]", num_proc=num_proc), load_dataset("mbpp", split="prompt[:3]", num_proc=num_proc))
    else:
        prompts = read_train_examples(load_dataset("mbpp", split="train", num_proc=num_proc), load_dataset("mbpp", split="prompt[:3]", num_proc=num_proc))

    model = args.model
    logger.info(f"loading model from {model}...")
    generator = pipeline('text-generation', model, device=0, batch_size=args.batch_size, torch_dtype='auto')
    logger.info(f"model loaded from {model}")

    set_seed(42)

    length = len(prompts)
    for sample in range(args.num_samples_per_task):
        generated_examples = [None] * length

        logger.info(f"generating sample {sample}...")
        generations = generator(prompts, return_full_text=False, **generate_kwargs)

        for i, generation in enumerate(generations):
            generated_examples[i] = {"sample": sample, "task_id": 601 + i, "generation": [convert_for_evaluation(generation[0]['generated_text'])]}

        if not generate_kwargs["do_sample"]:
            write_jsonl("data/mbpp/mbpp_compiler_feedback.jsonl", generated_examples)
            break

        if compiler == 'Cython':
            filename = command[1]
        elif compiler == 'Codon':
            filename = command[-1]
        else:
            raise ValueError

        index2new_prompt = {}

        for i, generated_example in enumerate(generated_examples):
            generation = generated_example["generation"][-1]
            with open(filename, 'w') as file:
                print(generation, file=file)
            output = run(command, capture_output=True)
            if output.returncode != 0:
                output = output.stderr.decode()[18:]
                try:
                    index2new_prompt[i] = prompts[i] + "```python\n" + '\n'.join(generation.splitlines()[:int(output[:output.find(':')]) - 1]) + '\n'
                except ValueError:
                    index2new_prompt[i] = prompts[i] + "```python\n"

        for attempt in range(1, args.num_attempts):
            generations = generator(index2new_prompt.values(), **generate_kwargs)

            for i, index in enumerate(index2new_prompt.keys()):
                generation = generations[i][0]['generated_text'][len(prompts[index]):]
                generated_examples[index]["generation"].append(convert_for_evaluation(generation))
                with open(filename, 'w') as file:
                    print(generation, file=file)
                output = run(command, capture_output=True)
                if output.returncode == 0:
                    index2new_prompt.pop(index)
                    continue
                output = output.stderr.decode()[18:]
                try:
                    index2new_prompt[i] = prompts[i] + "```python\n" + '\n'.join(
                        generation.splitlines()[:int(output[:output.find(':')]) - 1]) + '\n'
                except ValueError:
                    index2new_prompt[i] = prompts[i] + "```python\n"

        logger.info(f"generated sample {sample}")
        write_jsonl("data/mbpp/mbpp_compiler_feedback.jsonl", generated_examples, append=True)
