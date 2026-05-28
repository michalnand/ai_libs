from sktime.transformations.panel.catch22 import Catch22
import numpy

'''
CATCH22 : Classification and Regression with Time Series
'''
class Catch22Features:

    '''
        x_initial : numpy.array, sampled batch of shape : (num_samples, seq_len, num_features)
        num_kernels : int, number of random convolutional kernels to generate
    '''
    def __init__(self, x_initial):
        # Convert to sktime format: (num_samples, num_channels, seq_len)
        # Here, 'num_features' maps to 'num_channels'
        sktime_format = numpy.transpose(x_initial, (0, 2, 1))


        # Initialize Catch22
        self.c22 = Catch22()

        # Transform
        self.features = self.c22.fit_transform(sktime_format)

        z = self.forward(x_initial)

        self.num_features = z.shape[1]

    def forward(self, x):
        # Convert to sktime format: (num_samples, num_channels, seq_len)
        sktime_format = numpy.transpose(x, (0, 2, 1))
        
        features = self.c22.transform(sktime_format)

        z = features.to_numpy()
        z = numpy.array(z, dtype=numpy.float32)
        #z = numpy.nan_to_num(z)

        z[numpy.isnan(z)] = 0

        return z
    
    def __call__(self, x):
        return self.forward(x)
    

if __name__ == "__main__":

    num_samples = 10
    seq_len = 100
    num_features = 10

    x_initial = numpy.random.randn(num_samples, seq_len, num_features)

    catch = Catch22Features(x_initial)

    x_test = numpy.random.randn(num_samples, seq_len, num_features)

    z = catch.forward(x_test)
    print(z.shape)

    #print(z[0])
    #print(z[1])
    #print(z[2])
