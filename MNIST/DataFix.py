import pandas as pd
import csv
from keras.utils import np_utils

data = pd.read_csv("Results\mnist_keras.csv")

f = open("Results\mnist_keras_fix.csv", mode="w", newline="")
csvWriter = csv.writer(f)
csvWriter.writerow(["ImageId", "Label"])

for x in data[1:]:
    print(x)

f.close()