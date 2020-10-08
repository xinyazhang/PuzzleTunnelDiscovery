# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later
from __future__ import print_function

class Problem(object):

    def __init__(self):
        self.items = {}

    def _add_item(self, item):
        key = item.__class__.__name__
        if key not in self.items:
            self.items[key] = [item]
        else:
            self.items[key].append(item)

    def _add_unique_item(self, item):
        key = item.__class__.__name__
        self.items[key] = [item]

    def _sets(self, klass):
        return self.items[klass.__name__]

    def _print_commands(self):
        for fn in dir(self):
            if not fn.startswith('_'):
                print('\t'+fn)

    def _print_items(self):
        for key, item_list in self.items.items():
            print('{} {}:'.format(key, type(item).__name__), end=' ')
            for item in item_list:
                item.show()
                print()

    def _compose_condor_header(self):
        print('''+Group = "GRAD"
+Project = "GRAPHICS_VISUALIZATION"
+ProjectDescription = "Disentanglement Puzzle"
universe = vanilla
requirements = InMastodon''')


class ValidFile(object):

    def __init__(self, fn):
        if not os.path.isfile(fn):
            raise IOError("File {} not found".format(fn))
        self._fn = fn

#########################################
#                                       #
# THE FOLLOWING FUNCTIONS ARE TEMPLATES #
#                                       #
#########################################

def usage():
    print('''
#NAME# <command> [COMMAND ARGUMENTS]

Command list:''')
    pdo = NAME()
    for fn in dir(pdo):
        if not fn.startswith('_'):
            print('\t'+fn)

def main():
    cmd = sys.argv[1]
    if cmd in ['-h', '--help', 'help']:
        usage()
        return
    args = sys.argv[2:]
    try:
        pdo = pickle.load(open(PROBLEM_DEFINE_FILE, "rb"))
    except FileNotFoundError:
        pdo = ProblemDefine()

    if args:
        getattr(pdo, cmd)(args)
    else:
        getattr(pdo, cmd)()
    pickle.dump(pdo, open(PROBLEM_DEFINE_FILE, "wb"))
