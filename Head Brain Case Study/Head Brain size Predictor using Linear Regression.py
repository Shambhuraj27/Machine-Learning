
import pandas as pd
from sklearn.linear_model import LinearRegression

def HeadBrain():
    data = pd.read_csv("HeadBrain .csv")
    print("Size of Dataset",data.shape)

    X = data["Head Size(cm^3)"].values
    Y = data["Brain Weight(grams)"].values
    
    reg = LinearRegression()
    
    reg = reg.fit(X,Y)
    y_pred = reg.predict(X)
    
    r2 = reg.score(X,Y)
    print(r2)

def main():
    print("Head Brain Size Predictor using Linear regresion")
    HeadBrain()
if __name__ == "__main__":
    main()
