import pandas as pd
import math as math
import sklearn.svm as svm
import sklearn.cross_validation as crossVal
import sklearn.metrics as metrics
import numpy as np
import seaborn as sns
from sklearn.learning_curve import validation_curve

#Data Pre-processing
train_frame = pd.read_csv(r"C:\Users\Ishaan\PycharmProjects\Kaggle\titanic\data\train.csv", header = 0)
train_frame["Sex"] = train_frame["Sex"].map({"female" : 0, "male" : 1, "child" : 2}).astype(int)
train_frame = train_frame.drop("Cabin", axis=1)
train_frame = train_frame.drop("Name", axis=1)
train_frame = train_frame.drop("Ticket", axis=1)

sum = 0
count = 0
for i in train_frame["Age"]:
    if math.isnan(i) == False:
        count += 1
        sum += i
age_mean = sum/count

train_frame["Age"] = train_frame["Age"].fillna(int(age_mean))
train_frame["Age"] = train_frame["Age"].astype(int)
train_frame["Embarked"] = train_frame["Embarked"].fillna("S")
train_frame["Embarked"] = train_frame["Embarked"].map( {"S" : 0, "C" : 1, "Q" : 2}).astype(int)
train_data = train_frame.values

#Create SVMs
support_vector_machine = svm.SVC(C = 1,gamma=1e-8)
support_vector_machine_over = svm.SVC(C = 10000, gamma = 100)
support_vector_machine_under = svm.SVC(C = 1, gamma = 0.001)

#Train and test data for SVMs
train_x = train_data[:500, 2:]
train_x_over = train_data[:800, 2:]
train_x_under = train_data[:200, 2:]
train_y = train_data[:500, 1]
train_y_over = train_data[:800, 1]
train_y_under = train_data[:200, 1]
test_x = train_data[500:,2:]
test_x_over = train_data[800:, 2:]
test_x_under = train_data[200:, 2:]
test_y = train_data[500:, 1]
test_y_over = train_data[800:, 1]
test_y_under = train_data[200:, 1]

#Fit SVMs
support_vector_machine.fit(train_x, train_y)
support_vector_machine_over.fit(train_x_over, train_y_over)
support_vector_machine_under.fit(train_x_under, train_y_under)

#Cross validation Score
cv_score = crossVal.cross_val_score(support_vector_machine, train_x, train_y, cv = 5)
cv_score_over = crossVal.cross_val_score(support_vector_machine_over, train_x_over, train_y_over, cv = 5)
cv_score_under = crossVal.cross_val_score(support_vector_machine_under, train_x_under, train_y_under, cv = 5)

#Predict values
predict_y = support_vector_machine.predict(test_x)
predict_y_over = support_vector_machine_over.predict(test_x_over)
predict_y_under = support_vector_machine_under.predict(test_x_under)

#Accuracy
acc_score = metrics.accuracy_score(test_y, predict_y)
acc_score_over = metrics.accuracy_score(test_y_over, predict_y_over)
acc_score_under = metrics.accuracy_score(test_y_under, predict_y_under)

#Mean Absolute Error
m_a_e = metrics.mean_absolute_error(test_y, predict_y)
m_a_e_over = metrics.mean_absolute_error(test_y_over, predict_y_over)
m_a_e_under = metrics.mean_absolute_error(test_y_under, predict_y_under)

#Mean Square Error
m_s_e = metrics.mean_squared_error(test_y, predict_y)
m_s_e_over = metrics.mean_squared_error(test_y_over, predict_y_over)
m_s_e_under = metrics.mean_squared_error(test_y_under, predict_y_under)

#Normalised MSE
n_m_s_e = m_s_e/(max(test_y)-min(test_y))
n_m_s_e_over = m_s_e_over/(max(test_y_over)-min(test_y_over))
n_m_s_e_under = m_s_e_under/(max(test_y_under)-min(test_y_under))

#Print
print("Support Vector Machine Optimal: ",
      "\nCross Validation Score Max : ", cv_score.max(),
      "\nCross Vaildation Score Mean : ", cv_score.mean(),
      "\nTraining Accuracy : ", acc_score,
      "\nMean Absolute Error", m_a_e,
      "\nMean Squared Error : ", m_s_e,
      "\nNormalized MSE : ", n_m_s_e )

print("\nSupport Vector Machine Overfit: ",
      "\nCross Validation Score Max : ", cv_score_over.max(),
      "\nCross Vaildation Score Mean : ", cv_score_over.mean(),
      "\nTraining Accuracy : ", acc_score_over,
      "\nMean Absolute Error", m_a_e_over,
      "\nMean Squared Error : ", m_s_e_over,
      "\nNormalized MSE : ", n_m_s_e_over )

print("\nSupport Vector Machine Underfit: ",
      "\nCross Validation Score Max : ", cv_score_under.max(),
      "\nCross Vaildation Score Mean : ", cv_score_under.mean(),
      "\nTraining Accuracy : ", acc_score_under,
      "\nMean Absolute Error", m_a_e_under,
      "\nMean Squared Error : ", m_s_e_under,
      "\nNormalized MSE : ", n_m_s_e_under )

#Graph Section
param_range = np.logspace(-6, -2, 5)

train_scores, test_scores = validation_curve(
    support_vector_machine, train_x, train_y, param_name="gamma", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=1)

train_scores_over, test_scores_over = validation_curve(
    support_vector_machine_over, train_x_over, train_y_over, param_name="gamma", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=1)

train_scores_under, test_scores_under = validation_curve(
    support_vector_machine_under, train_x_under, train_y_under, param_name="gamma", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=1)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
train_scores_mean_over = np.mean(train_scores_over, axis=1)
train_scores_std_over = np.std(train_scores_over, axis=1)
test_scores_mean_over = np.mean(test_scores_over, axis=1)
test_scores_std_over = np.std(test_scores_over, axis=1)
train_scores_mean_under = np.mean(train_scores_under, axis=1)
train_scores_std_under = np.std(train_scores_under, axis=1)
test_scores_mean_under = np.mean(test_scores_under, axis=1)
test_scores_std_under = np.std(test_scores_under, axis=1)

sns.plt.figure(1)
sns.plt.subplot(131)
sns.plt.title("Validation Curve with SVM")
sns.plt.xlabel("$\gamma$")
sns.plt.ylabel("Score")
sns.plt.ylim(0.0, 1.1)
sns.plt.semilogx(param_range, train_scores_mean, label="Training score", color="r")
sns.plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2, color="r")
sns.plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="g")
sns.plt.fill_between(param_range, test_scores_mean - test_scores_std,
             test_scores_mean + test_scores_std, alpha=0.2, color="g")
sns.plt.legend(loc="best")
sns.plt.plot()

sns.plt.subplot(132)
sns.plt.title("Validation Curve with SVM OverFitting")
sns.plt.xlabel("$\gamma$")
sns.plt.ylabel("Score")
sns.plt.ylim(0.0, 1.1)
sns.plt.semilogx(param_range, train_scores_mean_over, label="Training score", color="r")
sns.plt.fill_between(param_range, train_scores_mean_over - train_scores_std_over,
                 train_scores_mean_over + train_scores_std_over, alpha=0.2, color="r")
sns.plt.semilogx(param_range, test_scores_mean_over, label="Cross-validation score",
             color="g")
sns.plt.fill_between(param_range, test_scores_mean_over - test_scores_std_over,
                 test_scores_mean_over + test_scores_std_over, alpha=0.2, color="g")
sns.plt.legend(loc="best")
sns.plt.plot()

sns.plt.subplot(133)
sns.plt.title("Validation Curve with SVM UnderFitting")
sns.plt.xlabel("$\gamma$")
sns.plt.ylabel("Score")
sns.plt.ylim(0.0, 1.1)
sns.plt.semilogx(param_range, train_scores_mean_under, label="Training score", color="r")
sns.plt.fill_between(param_range, train_scores_mean_under - train_scores_std_under,
                 train_scores_mean_under + train_scores_std_under, alpha=0.2, color="r")
sns.plt.semilogx(param_range, test_scores_mean_under, label="Cross-validation score",
             color="g")
sns.plt.fill_between(param_range, test_scores_mean_under - test_scores_std_under,
                 test_scores_mean_under + test_scores_std_under, alpha=0.2, color="g")
sns.plt.legend(loc="best")
sns.plt.plot()


figManager = sns.plt.get_current_fig_manager()
figManager.window.showMaximized()
sns.plt.savefig("Results-SVM/res_svm_graph2.png")
sns.plt.show()