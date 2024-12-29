
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report

def titanic():
    dataset = pd.read_csv("Titanic-Dataset.csv")
    print("First 5 rows of the dataset:")
    print(dataset.head())

    return dataset

def visualization(dataset):
    print("Survived and Non-survived passengers")
    sns.countplot(data=dataset,x="Survived",hue="Survived").set_title("Survived and Non-survived passengers")
    plt.legend(labels=["Non-survived","Survived"])
    plt.show()

    print("Survived and Non-survived passengers based on Passenger class")
    sns.countplot(data=dataset,x="Survived",hue="Pclass").set_title("Survived and Non-survived passengers based on Passenger class")
    plt.show()

    print("Survived and Non-survived passengers based on Gender")
    sns.countplot(data=dataset,x="Survived",hue="Sex").set_title("Survived and Non-survived passengers based on Gender")
    plt.legend(labels=["Male","Female"])
    plt.show()
    
    print("Survived and Non-survived passengers based on Age")
    dataset["Age"].plot.hist().set_title("Survived and Non-survived passengers based on Age")
    plt.show()

    print("Survived and Non-survived passengers based on Fare")
    dataset["Fare"].plot.hist().set_title("Survived and Non-survived passengers based on Fare")
    plt.show()
    print("-"*80)
    
def preprocess_data(dataset):
    # Find mising values
    print("Missing values in each column")
    print(dataset.isnull().sum())
    print("-"*80)

    # Handling missing values in 'Age' by filling with the mean
    print("Filling missing values in the 'Age' column with the mean value")
    dataset["Age"] = dataset["Age"].fillna(dataset["Age"].mean())
    print(dataset["Age"].head())
    print("-"*80)

    # Filling missing values in the 'Embarked' column with the mode value
    print("Filling missing values in the 'Embarked' column with the mode value")
    dataset["Embarked"] = dataset["Embarked"].fillna(dataset["Embarked"].mode()[0])
    print(dataset["Embarked"].head())
    print("-"*80)

    # Dropping irrelevant columns
    print("Dataset after dropping irrelevant columns")
    dataset.drop(columns=["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Cabin"],axis=1,inplace=True)
    print(dataset.head())
    print("-"*80)

    # Encoding categorical variables (Gender and Embarked)
    le = LabelEncoder()
    dataset["Sex"] = le.fit_transform(dataset["Sex"])
    print("Sex column after encoding\n",dataset["Sex"].head())
    print("-"*80)

    dataset["Embarked"] = le.fit_transform(dataset["Embarked"])
    print("Embarked column after encoding\n",dataset["Embarked"].head())
    print("-"*80)

    # Scaling numerical columns (excluding 'Survived')
    print("Scaling numerical columns")
    sc = StandardScaler()
    numerical_columns = dataset.select_dtypes(include=["int64","float64"]).columns
    numerical_columns = numerical_columns.drop("Survived")
    dataset[numerical_columns] = sc.fit_transform(dataset[numerical_columns])
    print("Preprocessing complete.")
    return dataset

def train_and_evaluate_model(x,y):
    # Splitting the dataset into training and testing sets
    print("Splitting the dataset into training and testing sets (70-30 split)")
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

    print("Training the Logistic Regression model")
    reg = LogisticRegression()
    reg.fit(x_train,y_train)

    print("Making predictions on the test set")
    prediction = reg.predict(x_test)

    # Calculating and displaying the accuracy score
    Accuracy = accuracy_score(y_test,prediction)
    print(f"Accuracy using Logistic Refression :{Accuracy * 100:.2f}%")
    print("-"*80)

    # Displaying the confusion matrix
    print("Confusion matrix:")
    cm = confusion_matrix(y_test,prediction)
    print(cm)
    print("-"*80)

    # Displaying the classification report
    print("Classification report:")
    cr = classification_report(y_test,prediction)
    print(cr)

def main():
    print("Titanic Survival Predictor")
    # Load the dataset
    dataset = titanic()

    print("Visualizing :")
    visualization(dataset=dataset)

    dataset = preprocess_data(dataset=dataset)

    # Separate features and target variable
    x = dataset.drop("Survived", axis=1)
    y = dataset["Survived"]
    
    train_and_evaluate_model(x,y)
    
if __name__ == "__main__":
    main()
