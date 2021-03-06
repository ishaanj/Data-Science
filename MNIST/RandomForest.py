import MNIST.DataClean as dc
import sklearn.ensemble as ensemble
import MLScripts.Helpers as helper
import MLScripts.Metrics as metrics
import csv

train_frame = dc.load_train_data()
train_data = dc.convertPandasDataFrameToNumpyArray(train_frame)
test_frame = dc.load_test_data()
test_data = dc.convertPandasDataFrameToNumpyArray(test_frame)

random_forest = ensemble.RandomForestClassifier(n_estimators=100, max_depth=4, random_state=1)
train_x = train_data[:, 1:]
train_y = train_data[:, 0]

random_forest.fit(train_x, train_y)
cv_score = metrics.crossValidationScore(random_forest, train_x, train_y, cvCount=5)

xTrain, xTest, yTrain, yTest = metrics.traintestSplit(train_x, train_y, randomState=1)
cv_forest = ensemble.RandomForestClassifier(max_depth=4, n_estimators=100, random_state=1)
cv_forest.fit(xTrain, yTrain)
y_predict = cv_forest.predict(xTest)
ta = metrics.trainingAccuracy(yTest, y_predict)
rmse = metrics.rmse(yTest, y_predict)
nrmse = metrics.nrmse(yTest, y_predict)

predictors = dc.getColNames(train_frame)[1:]
kfoldAccuracy = metrics.measureKFoldAccuracy(train_frame, random_forest, predictors, outputClass="label", outputClause="label", kFolds=10)

print("Max Cross Validation Score : ", cv_score.max(), "\nAverage Cross Validation Score : ", cv_score.mean(),
  "\nExtraTreeCLassifier Score : ", random_forest.score(xTrain, yTrain),
  "\nTraining Accuracy : ", ta,
  "\nRoot Mean Squared Error : ", rmse, "\nNormalized RMSE : ", nrmse,
  "\nKFold Accuracy : ", kfoldAccuracy)

featureNames = dc.getColNames(train_frame)[1:]
helper.printFeatureImportances(featureNames, random_forest.feature_importances_)

y_output = random_forest.predict(test_data[:, 0:])

f = open("mnist.csv", "w", newline="")
csvWriter = csv.writer(f)
csvWriter.writerow(["ImageId", "Label"])
i = 1
for pred_label in y_output:
    csvWriter.writerow([int(i), int(pred_label)])
    i = i+1
print("Number of Predictions: ", i)
f.close()
