PROMPT_TEMPLATE = '''Please refer the given examples and generate a python function for my problem.
Examples are listed as follows:
{}

Here is my problem:
{}
>>> Code:
'''

CYTHON_COMMAND = ("cython", "generationn.py", "-+", "--3")
CODON_COMMAND = ("codon", "build", "-release", "-llvm", "generationn.py")

generate_kwargs = {
    "do_sample": True,
    "temperature": 0.9,
    "top_k": 0,
    "top_p": 0.95
}
