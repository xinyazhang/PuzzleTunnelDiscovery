from __future__ import print_function
import threading
from six.moves import queue,input

QUEUE_CAPACITY = 16

_offloader = None
_Q = None
_train_writer = None

class Arguments:
    def __init__(self,
                 train_op,
                 summary=None,
                 global_step=None,
                 dic=None):
        self.train_op = train_op
        self.summary = summary
        self.global_step = global_step
        self.dic = dic

def worker():
    while True:
        a = _Q.get()
        if a.train_op is None:
            break

def init(args):
    global _offloader
    if _offloader is not None:
        return
    global _Q
    global _train_writer
    _Q = queue.Queue(QUEUE_CAPACITY)
    _offloader = threading.Thread(target=worker)
    _train_writer = train_writer

def offload(train_op, summary, global_step, dic):
    a = Arguments(train_op, summary, global_step, dic)
    _Q.put(a)

def down():
    _Q.put(Arguments(train_op=None))
    _offloader.join()
