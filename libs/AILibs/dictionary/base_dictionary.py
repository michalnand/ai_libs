"""
    AILibs Dictionaries: Compositional Computational Graph Architecture
    Supports 1:1, serial pipelines, and parallel stacking via a common base class.
"""
import numpy

class BaseDictionary:
    """
    Common ancestor for all dictionary nodes in the computational graph.
    Handles recursive execution of serial (single upstream) and parallel (list of upstream) nodes.
    """
    def __init__(self, upstream=None):
        self.upstream = upstream

    def __call__(self, x):
        x = numpy.atleast_2d(x)
        
        # 1. Resolve upstream dependencies recursively
        if self.upstream is None:
            z = x
        elif isinstance(self.upstream, (list, tuple)):
            # Parallel branch: compute all upstream nodes and hstack their outputs
            results = [d(x) for d in self.upstream]
            z = numpy.hstack(results) if results else x 
        elif isinstance(self.upstream, BaseDictionary):
            # Serial branch: compute single upstream node first
            z = self.upstream(x)
        else:
            raise TypeError("Upstream must be None, a BaseDictionary instance, or a list/tuple of dictionaries.")
        
        # 2. Apply current node's transformation
        return self._transform(z)

    def _transform(self, x):
        """Implemented by subclasses to perform the actual mathematical mapping."""
        raise NotImplementedError

    def __repr__(self):
        if self.upstream is None:
            return f"{self.__class__.__name__}()"
        elif isinstance(self.upstream, list):
            parents = ", ".join([str(d) for d in self.upstream])
            return f"{self.__class__.__name__}([{parents}])"
        else:
            return f"{self.__class__.__name__}({self.upstream})"