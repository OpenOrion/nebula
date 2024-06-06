from typing import Protocol, Union
from nebula.prim.sparse import SparseIndexable

class Topology(SparseIndexable, Protocol):

    def add(self, topology: "Topology", reorder_index: bool = False): ...
