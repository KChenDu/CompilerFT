import argparse

from datasets import Dataset, load_dataset
from setting import PROMPT_TEMPLATE, generate_kwargs
from re import search, DOTALL
from loguru import logger
from pathlib import Path
from os import cpu_count, remove
from transformers import pipeline, set_seed
from human_eval.data import write_jsonl
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
    parser.add_argument('--model', default='deepseek-ai/deepseek-coder-1.3b-instruct', type=str)
    parser.add_argument('--num_samples_per_task', default=20, type=int)
    parser.add_argument('--num_attempts', default=10, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--compiler', choices=('Cython', 'Codon'), default='Codon', type=str)
    parser.add_argument('--sample_offset', default=0, type=int)
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    compiler = args.compiler
    if compiler == 'Cython':
        from setting import CYTHON_COMMAND as command
        filename = Path(command[1])
    elif compiler == 'Codon':
        from setting import CODON_COMMAND as command
        filename = Path(command[-1])
    else:
        raise ValueError

    root = Path(__file__).parent

    num_proc = cpu_count()
    logger.info("loading MBPP dataset...")
    if args.demo:
        train_examples = load_dataset("mbpp", split="train[:16]", num_proc=num_proc)
    else:
        train_examples = load_dataset("mbpp", split="train", num_proc=num_proc)
    prompt_examples = load_dataset("mbpp", split="prompt[:3]", num_proc=num_proc)
    prompts = read_train_examples(train_examples, prompt_examples)
    length = len(prompts)
    logger.info(f"{length} problems loaded from MBPP dataset")

    model = args.model
    logger.info(f"loading model from {model}...")
    generator = pipeline('text-generation', model, device=0, batch_size=args.batch_size, torch_dtype='auto')
    logger.info(f"model loaded from {model}")

    set_seed(42)

    task_id_offset = train_examples[0]['task_id']
    sample_offset = args.sample_offset
    for sample in range(sample_offset, sample_offset + args.num_samples_per_task):
        generated_examples = [None] * length

        logger.info(f"generating sample {sample}...")
        generations = generator(prompts, max_new_tokens=1024, return_full_text=False, **generate_kwargs)

        for i, generation in enumerate(generations):
            generated_examples[i] = dict(sample=sample, task_id=task_id_offset + i, generation=[convert_for_evaluation(generation[0]['generated_text'])])

        if not generate_kwargs["do_sample"]:
            logger.info("generation over")
            write_jsonl(root / "mbpp_compiler_feedback.jsonl", generated_examples)
            exit()

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
            if len(index2new_prompt) < 1:
                break
            generations = generator(list(index2new_prompt.values()), max_new_tokens=1024, **generate_kwargs)

            new_index2new_prompt = {}

            for i, index in enumerate(index2new_prompt.keys()):
                generation = convert_for_evaluation(generations[i][0]['generated_text'][len(prompts[index]):])
                generated_examples[index]["generation"].append(generation)
                with open(filename, 'w') as file:
                    print(generation, file=file)
                output = run(command, capture_output=True)
                if output.returncode != 0:
                    output = output.stderr.decode()[18:]
                    try:
                        new_index2new_prompt[index] = prompts[index] + "```python\n" + '\n'.join(generation.splitlines()[:int(output[:output.find(':')]) - 1]) + '\n'
                    except ValueError:
                        new_index2new_prompt[index] = prompts[index] + "```python\n"
            index2new_prompt = new_index2new_prompt
        logger.info(f"sample {sample} generated")

        logger.info(f"saving sample {sample}...")
        write_jsonl(root / "mbpp_compiler_feedback.jsonl", generated_examples, append=True)
        logger.info(f"sample {sample} saved")

    if compiler == "Cython":
        remove(filename.with_suffix(".cpp"))
    elif compiler == 'Codon':
        remove(filename.with_suffix(".ll"))
    else:
        raise ValueError
    remove(filename)
