from MLScripts.CleaningUtils import *

def load_train_data(desc=False):
    return loadData(r"C:\Users\Ishaan\PycharmProjects\Kaggle\MNIST\Data\train.csv", describe=desc)

def load_test_data(desc=False):
    return loadData(r"C:\Users\Ishaan\PycharmProjects\Kaggle\MNIST\Data\test.csv", describe=desc)


