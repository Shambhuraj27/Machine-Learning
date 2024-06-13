
import pandas as pd
from seaborn import countplot
from matplotlib.pyplot import figure,show
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def TitanicLogistic():
    # Step 1 : Load data
    titanic_data = pd.read_csv("Titanic.csv")

    print("First 5 entries from loaded dataset")
    print(titanic_data.head())
    print("Number of passengers are :",(len(titanic_data)))
    print("---------------------------------------------------")
    # Step 2 : Analyze data
    print("Visualisation : Survived and non survived passengers")
    figure() 
    target = "Survived" 
    countplot(data=titanic_data,x=target).set_title("Survived and non survived passengers") # Used to show the counts of observations in each categorical bin using bars.
    show() 

    print("Visualisation : Survived and non survived passengers based on the Gender")
    figure()
    target = "Survived"
    countplot(data=titanic_data,x=target,hue="Sex").set_title("Survived and non survived passengers based on the Gender")
    show()

    print("Visualisation : Survived and non survived passengers based on the Passenger class")
    figure()
    target = "Survived"
    countplot(data=titanic_data,x=target,hue="Pclass").set_title("Survived and non survived passengers based on Passenger class")
    show()

    print("Visualisation : Survived and non survived passengers based on Age")
    figure()
    titanic_data["Age"].plot.hist().set_title("Survived and non survived passengers based on Age")
    show()

    print("Visualisation : Survived and non survived passengers based on the Fare")
    figure()
    titanic_data["Fare"].plot.hist().set_title("Visualisation : Survived and non survived passengers based on the Fare")
    show()

    # Step 3 : Data Cleaning
    titanic_data.drop("zero",axis = 1,inplace = True)

    print("First 5 entries from loaded dataset after removing zero column")
    print(titanic_data.head(5))

    # Convert categorical variables into dummy variables
    print("Values of Sex column")
    print(pd.get_dummies(titanic_data["Sex"]))

    # Drop one dummy variable to avoid multicollinearity
    print("Values of Sex column after removing one field")
    Sex = pd.get_dummies(titanic_data["Sex"],drop_first = True)
    print(Sex.head(5))

    print("Values of Pclass column after removing one field ")
    Pclass = pd.get_dummies(titanic_data["Pclass"],drop_first=True)
    print(Pclass.head(5))

    print("Values of dataset after concatenating new columns")
    titanic_data = pd.concat([titanic_data,Sex,Pclass],axis = 1)
    print(titanic_data.head(5))

    print("Values of dataset after removing irrelevant columns")
    titanic_data.drop(["Sex","sibsp","Parch","Embarked"],axis = 1,inplace = True)
    print(titanic_data.head(5))

    x = titanic_data.drop("Survived",axis = 1)
    y = titanic_data["Survived"]

    # Step 4 : Data Training
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.5)

    xtrain.columns = xtrain.columns.astype(str)
    xtest.columns = xtest.columns.astype(str)

    logmodel = LogisticRegression(max_iter=1000)
    logmodel.fit(xtrain,ytrain)

    # Step 5 : Data Testing
    prediction = logmodel.predict(xtest)

    # Step 5 : Calculate Accuracy
    print("Accuracy of dataset using Logistic Regression is :")
    print(accuracy_score(ytest,prediction))

def main():
    print("Supervised Machine Learning")
    print("Logistic Regression on Titanic Dataset")
    TitanicLogistic()
if __name__ == "__main__":
    main()
