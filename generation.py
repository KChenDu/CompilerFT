from heapq import heappop, heappush

class Pair(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.len = b - a

    def __lt__(self, other):
        return self.len < other.len

    def __str__(self):
        return '({},{})'.format(self.a, self.b)

def max_chain_length(pairs, n):
    if len(pairs) == 0 or len(pairs) == 1:
        return len(pairs)

    # Initialize result
    result = 0

    # sort the pairs based on increasing order of their first element
    pairs.sort(key=lambda x: x.a)

    # Initialize max end of last added chain with the
    # first of the last element in pairs.
    max_end = pairs[0].b

    # Go through all the remaining pairs
    for i in range(1, len(pairs)):
        if pairs[i].a > max_end:
            # If new interval begins with the max end of
            # previous chain, we can add it to the current
            # result
            result += 1
            max_end = pairs[i].b

    return result

