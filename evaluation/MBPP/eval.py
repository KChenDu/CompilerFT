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
from human_eval.evaluation import evaluate_functional_correctness


def read_test_examples(train_examples: Dataset, prompt_examples: Dataset) -> list[str]:
    def format_test_example(q: str, tests: list[str], code: str | None = None):
        prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}".format(q, '\n'.join(map(str.strip, tests)))
        if code is not None:
            prompt += f"\n>>> Code:\n```python\n{code.strip()}\n```"
        return prompt

    examples_str = [None, None, None]

    for i in range(3):
        example_prompt = format_test_example(prompt_examples[i]['text'], prompt_examples[i]['test_list'], prompt_examples[i]['code'])
        examples_str[i] = f'- Example {i + 1}:\n{example_prompt}'

    prompts = [None] * len(train_examples)

    for i, example in enumerate(train_examples):
        prompt = format_test_example(example['text'], example['test_list'])
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
    parser.add_argument('--num_attempts', default=5, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--compiler', choices=('Cython', 'Codon'), default='Codon', type=str)
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

    logger.info("loading MBPP dataset...")
    num_proc = cpu_count()
    test_examples = load_dataset("mbpp", split="test", num_proc=num_proc)
    prompt_examples = load_dataset("mbpp", split="prompt[:3]", num_proc=num_proc)
    prompts = read_test_examples(test_examples, prompt_examples)
    length = len(prompts)
    logger.info(f"{length} problems loaded from MBPP dataset")

    model = args.model
    logger.info(f"loading model from {model}...")
    generator = pipeline('text-generation', model, device=0, batch_size=args.batch_size, torch_dtype='auto')
    logger.info(f"model loaded from {model}")

    logger.info("generating...")
    set_seed(42)
    generated_examples = [None] * length
    generations = generator(prompts, max_new_tokens=512, return_full_text=False, **generate_kwargs)

    task_id_offset = test_examples[0]['task_id']
    for i, generation in enumerate(generations):
        generated_examples[i] = dict(task_id=task_id_offset + i, generation=convert_for_evaluation(generation[0]['generated_text']))
    logger.info("generation over")
    
    index2new_prompt = {}

    for i, generated_example in enumerate(generated_examples):
        generation = generated_example["generation"]
        with open(filename, 'w') as file:
            print(generation, file=file)
        output = run(command, capture_output=True)
        if output.returncode != 0:
            output = output.stderr.decode()[18:]
            try:
                index2new_prompt[i] = prompts[i] + "```python\n" + '\n'.join(generation.splitlines()[:int(output[:output.find(':')]) - 1]) + '\n'
            except ValueError:
                index2new_prompt[i] = prompts[i] + "```python\n"

    print("compilability:", (length - len(index2new_prompt)) / length)
    root = Path(__file__).parent
    write_jsonl(root / "mbpp_samples.jsonl", generated_examples)
    result, results = evaluate_functional_correctness(str(root / "mbpp_samples.jsonl"), problem_file=str(root / "data" / "mbpp_test.jsonl"), is_mbpp=True)
    print("MBPP test score:", result)
    for r in results:
        if r[0][1]["passed"] and r[0][1]["task_id"] - task_id_offset in index2new_prompt:
            index2new_prompt.pop(r[0][1]["task_id"] - task_id_offset)
    
    if generate_kwargs["do_sample"]:
        for attempt in range(1, args.num_attempts):
            logger.info("regenerating...")
            if len(index2new_prompt) < 1:
                break
            generations = generator(list(index2new_prompt.values()), max_new_tokens=512, **generate_kwargs)

            new_index2new_prompt = {}

            for i, index in enumerate(index2new_prompt.keys()):
                generation = convert_for_evaluation(generations[i][0]['generated_text'][len(prompts[index]):])
                generated_examples[index]["generation"] = generation
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
            logger.info("regeneration over")
            write_jsonl(root / "mbpp_samples.jsonl", generated_examples)
            _, results = evaluate_functional_correctness(str(root / "mbpp_samples.jsonl"), problem_file=str(root / "data" / "mbpp_test.jsonl"), is_mbpp=True)
            for r in results:
                if r[0][1]["passed"] and r[0][1]["task_id"] - task_id_offset in index2new_prompt:
                    index2new_prompt.pop(r[0][1]["task_id"] - task_id_offset)

        if compiler == "Cython":
            remove(filename.with_suffix(".cpp"))
        elif compiler == 'Codon':
            remove(filename.with_suffix(".ll"))
        else:
            raise ValueError
        remove(filename)

    logger.info("saving {} processed examples into {}...".format(length, root / "mbpp_samples.jsonl"))
    write_jsonl(root / "mbpp_samples.jsonl", generated_examples)
    logger.info("saved {} processed examples into {}".format(length, root / "mbpp_samples.jsonl"))

    result, _ = evaluate_functional_correctness(str(root / "mbpp_samples.jsonl"), problem_file=str(root / "data" / "mbpp_test.jsonl"), is_mbpp=True)
    print("compilability:", (500 - len(index2new_prompt)) / 500)
    print("MBPP test score:", result)
