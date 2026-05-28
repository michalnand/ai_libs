from sktime.transformations.panel.rocket import Rocket, MiniRocket
import numpy

'''
ROCKET : RandOm Convolutional KErnel Transform

variants : Rocket, MiniRocket, MultiRocket
'''
class RocketFeatures:

    '''
        x_initial : numpy.array, sampled batch of shape : (num_samples, seq_len, num_features)
        num_kernels : int, number of random convolutional kernels to generate

        minirocker produces 84 x num_kernels features per input channel
    '''
    def __init__(self, x_initial, num_kernels = 128):
        # Convert to sktime format: (num_samples, num_channels, seq_len)
        # Here, 'num_features' maps to 'num_channels'
        sktime_format = numpy.transpose(x_initial, (0, 2, 1))

        # initialisation
        self.rocket = MiniRocket(num_kernels=num_kernels) 
        self.rocket.fit(sktime_format)

        z = self.forward(x_initial)

        self.num_features = z.shape[1]

    def forward(self, x):
        # Convert to sktime format: (num_samples, num_channels, seq_len)
        sktime_format = numpy.transpose(x, (0, 2, 1))
        
        features = self.rocket.transform(sktime_format)

        return features.to_numpy()
    
    def __call__(self, x):
        return self.forward(x)
    

if __name__ == "__main__":

    num_samples = 10
    seq_len = 100
    num_features = 10

    x_initial = numpy.random.randn(num_samples, seq_len, num_features)

    rocket = RocketFeatures(x_initial, num_kernels=128)

    x_test = numpy.random.randn(num_samples, seq_len, num_features)

    z = rocket.forward(x_test)
    print(z.shape)

    #print(z[0])
    #print(z[1])
    #print(z[2])
