import numpy
from PIL import Image


class SegmentationAugmentation:


    def __init__(self):
        self.crop_max = 0.4
        self.angle_max = 30.0


    def __call__(self, x, y):

        # flip
        if numpy.random.rand() > 0.5:
            x = numpy.flip(x, axis=1)
            y = numpy.flip(y, axis=0)

        if numpy.random.rand() > 0.5:
            x = numpy.flip(x, axis=2)
            y = numpy.flip(y, axis=1)

        # random crop
        if numpy.random.rand() > 0.5:
            h = x.shape[1]  
            w = x.shape[2]

            x_min = numpy.random.randint(0, int(w*self.crop_max))
            y_min = numpy.random.randint(0, int(h*self.crop_max))

            x_max = w - numpy.random.randint(0, int(w*self.crop_max))
            y_max = h - numpy.random.randint(0, int(h*self.crop_max))

            rect = (x_min, y_min, x_max, y_max)


            x = numpy.moveaxis(x, 0, 2)
            img = Image.fromarray(numpy.uint8(x*255))
            x = img.crop(rect)
            x = x.resize((w, h))
            x = numpy.moveaxis(numpy.array(x)/255.0, 2, 0)

            img = Image.fromarray(numpy.uint8(y))
            y = img.crop(rect)
            y = y.resize((w, h))
            y = numpy.array(y, dtype=int)

        # random rotation
        if numpy.random.rand() > 0.5:
            angle = numpy.random.uniform(-self.angle_max, self.angle_max)

            x = numpy.moveaxis(x, 0, 2)
            img = Image.fromarray(numpy.uint8(x*255))
            x = img.rotate(angle)
            x = numpy.moveaxis(numpy.array(x)/255.0, 2, 0)

          
            img = Image.fromarray(numpy.uint8(y))
            y = img.rotate(angle)
            y = numpy.array(y, dtype=int)

        

        # contrast and brightness
        if numpy.random.rand() > 0.5:
            x = x*numpy.random.uniform(0.1, 2.0)
        
        if numpy.random.rand() > 0.5:
            x = x + numpy.random.uniform(-0.5, 0.5)

        # add noise
        if numpy.random.rand() > 0.5:
            k = numpy.random.uniform(0.05, 0.25)
            x = x + k*numpy.random.randn(x.shape[0], x.shape[1], x.shape[2])

        # negative image
        if numpy.random.rand() > 0.5:
            x = 1.0 - x

        if numpy.random.rand() > 0.5:
            x[0] = 1.0 - x[0]

        if numpy.random.rand() > 0.5:
            x[1] = 1.0 - x[1]

        if numpy.random.rand() > 0.5:
            x[2] = 1.0 - x[2]   
        
        # random channel shuffling
        if numpy.random.rand() > 0.0:
            perm = numpy.random.permutation(3)
            x = x[perm, :, :]
        


        x = numpy.clip(x, 0.0, 1.0)
        return numpy.array(x), numpy.array(y)
