import numpy


class DatasetBalancer:  

    def __init__(self, dataset, num_classes):
        self.class_indices = {}

        for n in range(num_classes):
            self.class_indices[n] = {"class_id" : n, "indices" : []}

        for n in range(len(dataset)):
            _, class_label = dataset[n]
            self.class_indices[class_label]["indices"].append(n)

        for n in range(len(self.class_indices)):
            class_id = self.class_indices[n]["class_id"]
            count    = len(self.class_indices[n]["indices"])
            print("class ", class_id,  " count :", count)


    def sample(self):
        # uniform sample random class
        class_label = numpy.random.randint(0, len(self.class_indices))

        # some dataset may have zero count for given class
        while len(self.class_indices[class_label]["indices"]) == 0:
            class_label = numpy.random.randint(0, len(self.class_indices))

        idx = numpy.random.choice(self.class_indices[class_label]["indices"])

        return idx


      