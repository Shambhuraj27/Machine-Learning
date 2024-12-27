
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def wine_predictor():
    dataset = pd.read_csv("WinePredictor.csv")
    print(dataset.head())

    x = dataset.drop("Class",axis=1)
    y = dataset["Class"]
    
    # Feature scaling for scale the training features
    sc = StandardScaler()
    print(sc.fit(x))
    print(sc.transform(x))
    x = pd.DataFrame(sc.transform(x),columns=x.columns)
    print(x.head())
    
    # Visualization 
    print("Visualizing pairplot...")
    sns.pairplot(dataset, hue='Class', diag_kind='kde', markers=["o", "s", "D"])
    plt.title("Pairplot of Features")
    plt.show()
    
    # Split the data into training and testing
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=42)

    reg = LogisticRegression(max_iter=10000,multi_class="multinomial",class_weight="balanced")
    
    # Train the model
    reg.fit(xtrain,ytrain)
    
    # Test the model
    prediction = reg.predict(xtest)
    
    # Calculate accuracy
    Accuracy = accuracy_score(ytest,prediction)
    print("Accuracy using Logistic Regression :",Accuracy*100,"%")

    # Custom input
    custom_input =[[13.24,2.59,2.87,21.0,118,2.80,2.69,0.39,1.82,4.32,1.04,2.93,735]]
    
    # Scale custom input for prediction
    custom_input_scaled = sc.transform(custom_input)
    custom_prediction = reg.predict(custom_input_scaled)
    print("Prediction for the provided custom input:", custom_prediction[0])

def main():
    print("Wine predictor")
    wine_predictor()
if __name__ == "__main__":
    main()
