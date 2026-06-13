import numpy

class BatchSampler:
    # inherit from this class any other sampler

    def sample(self, batch_size: int) -> numpy.ndarray | None:
        return None
        