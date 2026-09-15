from . import image_dataset

def get_datasets_valid(opts):
    trainset = image_dataset.Dataset_Train(opts)
    validset = image_dataset.Dataset_Valid(opts)

    return trainset, validset


def get_datasets_test(opts):
    trainset = image_dataset.Dataset_Train(opts)
    testset = image_dataset.Dataset_Test(opts)

    return trainset, testset