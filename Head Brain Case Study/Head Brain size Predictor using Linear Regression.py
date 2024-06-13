
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def HeadBrain():
    data = pd.read_csv("HeadBrain .csv")
    print("Size of Dataset",data.shape)

    X = data["Head Size(cm^3)"].values
    Y = data["Brain Weight(grams)"].values

    X = X.reshape((-1,1))

    reg = LinearRegression()
    reg = reg.fit(X,Y)
    y_pred = reg.predict(X)

    plt.scatter(X, Y, color="Orange")
    plt.plot(X, y_pred, color="Green")
    plt.xlabel("Head Size (cm^3)")
    plt.ylabel("Brain Weight (grams)")
    plt.show()

    r2 = reg.score(X,Y)
    print("Accuracy of Linear Regression is :",r2*100)

def main():
    print("Head Brain Size Predictor using Linear regression")
    HeadBrain()
if __name__ == "__main__":
    main()
