import titanic.DataClean as dc
import sklearn.ensemble as ensemble
import MLScripts.Metrics as metrics
import MLScripts.Helpers as helper
import csv

train_frame = dc.dfCleanData(dc.loadTrainData(desc=False))
dc.describeDataframe(train_frame)
train_data = dc.convertPandasDataFrameToNumpyArray(train_frame)
test_frame = dc.dfCleanDataTest(dc.loadTestData())
test_data = dc.convertPandasDataFrameToNumpyArray(test_frame)
dc.describeDataframe(test_frame)

random_forest = ensemble.RandomForestClassifier(max_depth=3, n_estimators=100, random_state=0)
train_x = train_data[:, 1:]
train_y = train_data[:, 0]

random_forest.fit(train_x, train_y)
cv_score = metrics.crossValidationScore(random_forest, train_x, train_y, cvCount=5)

xTrain, xTest, yTrain, yTest = metrics.traintestSplit(train_x, train_y, randomState=1)
cv_forest = ensemble.RandomForestClassifier(max_depth=3, n_estimators=100, random_state=0)
cv_forest.fit(xTrain, yTrain)
y_predict = cv_forest.predict(xTest)
ta = metrics.trainingAccuracy(yTest, y_predict)
rmse = metrics.rmse(yTest, y_predict)
nrmse = metrics.nrmse(yTest, y_predict)

predictors = dc.getColNames(train_frame)[1:]
kfoldAccuracy = metrics.measureKFoldAccuracy(train_frame, random_forest, predictors, outputClass="Survived", outputClause="Survived", kFolds=10)

print("Max Cross Validation Score : ", cv_score.max(), "\nAverage Cross Validation Score : ", cv_score.mean(),
  "\nExtraTreeCLassifier Score : ", random_forest.score(xTrain, yTrain),
  "\nTraining Accuracy : ", ta,
  "\nRoot Mean Squared Error : ", rmse, "\nNormalized RMSE : ", nrmse,
  "\nKFold Accuracy : ", kfoldAccuracy)

featureNames = dc.getColNames(train_frame)[1:]
helper.printFeatureImportances(featureNames, random_forest.feature_importances_)

final_y_pred = random_forest.predict(test_data[:, 1:])

f = open("titanic-result.csv", "w", newline="")
csvWriter = csv.writer(f)
csvWriter.writerow(["PassengerId", "Survived"])
for pid, survive in zip(test_data[:, 0], final_y_pred):
  csvWriter.writerow([int(pid), int(survive)])
f.close()