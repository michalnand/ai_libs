import numpy
import AILibs
import matplotlib.pyplot as plt



if __name__ == "__main__":
    file_name = "/Users/michal/datasets/creditcard/creditcard.csv"

    dataset_orig  = AILibs.datasets.CSVDataset(file_name)

    print("dataset = ", dataset_orig.x.shape)

    # random dataset split, train test split, 80% train, 20% test
    test_ratio = 0.2
    indices = numpy.arange(dataset_orig.x.shape[0])
    numpy.random.shuffle(indices)

    split_idx       = int((1 - test_ratio) * dataset_orig.x.shape[0])
    
    train_indices   = indices[:split_idx]
    test_indices    = indices[split_idx:]

    x_train = dataset_orig.x[train_indices, :-1]

    x_test  = dataset_orig.x[test_indices, :-1]

    # target indficator is in the last column, and is only for evaluation, not for training
    y_gt    = dataset_orig.x[test_indices, -1]


    print("dataset train = ", x_train.shape)
    print("dataset test  = ", x_test.shape)
    print("\n\n")   


    print("Fitting Isolation Forest...")

    forest = AILibs.forest.IsolationForest()
    #forest = AILibs.forest.ExtendedIsolationForest()
    forest.fit(x_train, max_depth=12, num_trees=128, num_subsamples=4096)

    print("Predicting with Isolation Forest...")
    scores = forest.predict(x_test)


    th = AILibs.metrics.tune_threshold(y_gt, scores, metric="f1")
    metrics = AILibs.metrics.anomaly_evaluation(y_gt, scores, th=th)

    print("\n\n")
    print(AILibs.metrics.format_metrics(metrics))
    