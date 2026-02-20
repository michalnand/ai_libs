import numpy

class TestingDataset:   
    def __init__(self, num_samples_a = 1000, num_samples_b = 100, n_dims = 2):
        xa = self._create_samples(num_samples_a, n_dims, 1.0)
        xb = self._create_samples(num_samples_b, n_dims, 10.0)

        self.x = numpy.vstack([xa, xb])

        self.y  = numpy.zeros((num_samples_a + num_samples_b, 1), dtype=int)
        self.y[num_samples_a:] = 1


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


    def _create_samples(self, count, n_dims, scale_proj = 1.0):
        proj        = scale_proj*numpy.random.randn(n_dims+1, n_dims+1)
        
        x           = numpy.random.randn(count, n_dims+1)
        x[:, -1]    = 1.0

        result      = x@proj
        result      = result[:, 0:n_dims]

        return result