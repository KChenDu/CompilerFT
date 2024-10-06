PROMPT_TEMPLATE = '''Please refer the given examples and generate a python function for my problem.
Examples are listed as follows:
{}

Here is my problem:
{}
>>> Code:
'''

CYTHON_COMMAND = ("cython", "generationn.py", "-+", "--3")
CODON_COMMAND = ("codon", "build", "-release", "-llvm", "generationn.py")
