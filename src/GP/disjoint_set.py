#!/usr/bin/python

class DisjointSet:

    def __init__(self, init_arr):
        self._parents = {}
        if init_arr:
            for item in init_arr:
                self._parents[item] = item

    def find(self, elem):
        if elem not in self._parents:
            return None
        path = []
        while self._parents[elem] != elem:
            path.append(elem)
            elem = self._parents[elem]
        # path compression
        for e in path[1:]:
            self._parents[e] = elem
        return elem

    def union(self, elem1, elem2):
        if elem1 == elem2:
            return
        # TODO: more advanced algorithm ?
        elem1 = self.find(elem1)
        elem2 = self.find(elem2)
        assert elem1 is not None and elem2 is not None
        self._parents[elem1] = elem2

    def get_roots(self):
        roots = []
        for key in self._parents:
            value = self._parents[key]
            if value == key:
                roots.append(key)
        return roots
