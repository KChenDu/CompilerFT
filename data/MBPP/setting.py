PROMPT_TEMPLATE = '''Please refer the given examples and generate a Python function for my problem.
Examples are listed as follows:
{}

Here is my problem:
{}
>>> Code:
'''

CYTHON_COMMAND = ("cython", "generation.py", "-+", "--3")
CODON_COMMAND = ("codon", "build", "-release", "-llvm", "generation.py")

generate_kwargs = {
    "do_sample": True,
    "temperature": 0.7,
    "max_new_tokens": 1024,
    "top_k": 0,
    "top_p": 0.95
}
