import numpy as np
'''
partt.py -- a library that PARTitions Tasks

'''

def chunk_it(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out

def guess_chunk_number(shape, max_chunk_number, minimal_chunk_size):
    ntask = np.prod(shape)
    bound1 = max_chunk_number
    bound2 = ntask / float(minimal_chunk_size)
    return max(1, min(int(bound1), int(bound2)))

# A foolproof task partitioner
# May be slow, but we are not gonna have billions of tasks
def get_task_partition(shape, total_chunks):
    grand = [e for e in np.ndindex(shape)]
    return chunk_it(grand, total_chunks)

def get_task_chunk(shape, total_chunks, index):
    return get_task_allocation(shape, total_chunks)[index]
