
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

def PlayPredictor(data_path):
    data = pd.read_csv(data_path,index_col=0)
    print("Size of dataset :",len(data))

    feature_names = ["Whether","Temperature"]
    print("Names of feature are ",feature_names)

    Whether = data.Whether
    Temperature = data.Temperature
    Play = data.Play

    le = preprocessing.LabelEncoder()

    Whether_encoder = le.fit_transform(Whether)
    print(Whether_encoder)

    Temperature_encoder = le.fit_transform(Temperature)
    label = le.fit_transform(Play)
    print(Temperature_encoder)

    features = list(zip(Whether_encoder,Temperature_encoder))

    Model = KNeighborsClassifier()

    Model.fit(features,label)
    Predicted = Model.predict([[0,2]])
    print(Predicted)

def main():
    print("Play Predictor application using K Nearest Neighbor")
    PlayPredictor("PlayPredictor.csv")
if __name__ == "__main__":
    main()




