import titanic.DataClean as dc
import sklearn.tree as tree
import MLScripts.Metrics as metrics
import MLScripts.Helpers as helper
import csv

train_frame = dc.dfCleanData(dc.loadTrainData(desc=False))
train_data = dc.convertPandasDataFrameToNumpyArray(train_frame)
test_frame = dc.dfCleanDataTest(dc.loadTestData())
test_data = dc.convertPandasDataFrameToNumpyArray(test_frame)
dc.describeDataframe(test_frame)

decision_tree = tree.DecisionTreeClassifier(max_depth=4)
train_x = train_data[:, 1:]
train_y = train_data[:, 0]

decision_tree.fit(train_x, train_y)
cv_score = metrics.crossValidationScore(decision_tree, train_x, train_y, cvCount=5)

xTrain, xTest, yTrain, yTest = metrics.traintestSplit(train_x, train_y, randomState=1)
cv_tree = tree.DecisionTreeClassifier(max_depth=4)
cv_tree.fit(xTrain,yTrain)
y_predict = cv_tree.predict(xTest)
ta = metrics.trainingAccuracy(yTest, y_predict)
rmse = metrics.rmse(yTest, y_predict)
nrmse = metrics.nrmse(yTest, y_predict)

predictors = dc.getColNames(train_frame)[1:]
kfoldAccuracy = metrics.measureKFoldAccuracy(train_frame, decision_tree, predictors, outputClass="Survived", outputClause="Survived", kFolds=10)

print("Max Cross Validation Score : ", cv_score.max(), "\nAverage Cross Validation Score : ", cv_score.mean(),
  "\nExtraTreeCLassifier Score : ", decision_tree.score(xTrain, yTrain),
  "\nTraining Accuracy : ", ta,
  "\nRoot Mean Squared Error : ", rmse, "\nNormalized RMSE : ", nrmse,
  "\nKFold Accuracy : ", kfoldAccuracy)

featureNames = dc.getColNames(train_frame)[1:]
helper.printFeatureImportances(featureNames, decision_tree.feature_importances_)

final_y_pred = decision_tree.predict(test_data[:, 1:])

f = open("titanic-result.csv", "w", newline="")
csvWriter = csv.writer(f)
csvWriter.writerow(["PassengerId", "Survived"])
for pid, survive in zip(test_data[:, 0], final_y_pred):
  csvWriter.writerow([int(pid), int(survive)])
f.close()