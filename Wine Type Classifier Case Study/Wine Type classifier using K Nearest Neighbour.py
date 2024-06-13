
from sklearn import metrics
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def WinePredictor():
    # Load Dataset
    Wine = datasets.load_wine()

    # print the names of features
    print(Wine.feature_names)

    # print the label species(class_0,class_1,class_2)
    print(Wine.target_names)

    # print the wine data(top 5 records)
    print(Wine.target[0:5])

    # print the wine labels(0:class_0,1:class_1,2:class_2)
    print(Wine.target)

    x_train,x_test,y_train,y_test = train_test_split(Wine.data,Wine.target,test_size=0.3) #70% for training and 30% for testing

    # Create KNN classifir
    knn = KNeighborsClassifier(n_neighbors=3)

    knn.fit(x_train,y_train)
    knn.predict(x_test)

    print("Accuracy:",metrics.accuracy_score(y_test,knn.predict(x_test)))

def main():
    print("Wine Predictor application using KNN algorithn")
    WinePredictor()
if __name__ == "__main__":
    main()

