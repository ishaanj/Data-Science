from MLScripts.CleaningUtils import *

def loadTrainData(desc=False):
    df = loadData(r"C:\Users\Ishaan\PycharmProjects\Kaggle\titanic\data\train.csv", describe=desc)
    return df

def mapAge(df):
    copyFeatures(df, "Age", "AgeFill")
    df["Gender"] = df["Sex"].map( {"female" : 0, "male" : 1, "child" : 2}).astype(int)
    medianAges = np.zeros((2, 3))
    for i in range(2):
        for j in range(3):
            medianAges[i,j] = df[(df["Gender"] == i) & (df["Pclass"] == (j+1))]["Age"].dropna().mean()
    for i in range(2):
        for j in range(3):
            df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == (j+1)), "AgeFill"] = medianAges[i,j]


def loadTestData(desc=False):
    return loadData(r"C:\Users\Ishaan\PycharmProjects\Kaggle\titanic\data\test.csv", describe=desc)

def dfCleanData(df, istest=False):
    mapAge(df)
    addFeatures(df)
    df = dropUnimportantFeatures(df, ["PassengerId", "Name", "Sex", "Ticket", "Embarked", "Cabin", "SibSp", "Age"])
    if istest: df["Fare"] = df["Fare"].fillna(df["Fare"].mean)
    return df

def dfCleanDataTest(df):
    mapAge(df)
    df["Fare"] = df["Fare"].fillna(df["Fare"].mean())
    addFeatures(df)
    df = dropUnimportantFeatures(df, ["Name", "Sex", "Ticket", "Embarked", "Cabin", "SibSp", "Age"])
    return df

def addFeatures(df):
    df["FamilySize"] = df["SibSp"] + df["Parch"]
    df["Age*Class"] = df["AgeFill"] * df["Pclass"]
    df["FarePerPerson"] = df["Fare"] / ((df["FamilySize"] + 1))