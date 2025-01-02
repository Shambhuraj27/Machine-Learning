
import pandas as pd
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

def weather():
    # Load the data
    dataset = pd.read_csv("weather_data.csv")
    print(dataset.shape)
    # Display the first few rows and check for missing values
    print("First 5 rows from the loaded dataset :")
    print(dataset.head())
    print("-"*140)
    return dataset

def preprocess_data(dataset):
    print("Find missing values :")
    print(dataset.isnull().sum())
    print("-"*140)

    # Concatenate the encoded columns back into the dataset and drop the original 'Location' column
    print("Location column after encoding :")
    encoded_column = pd.get_dummies(dataset["Location"], drop_first=True)
    print("-"*140)

    print("Dataset after concatenating columns")
    dataset = pd.concat([dataset, encoded_column], axis=1)
    dataset = dataset.drop(columns=["Location"])
    print(dataset.head())
    print("-"*140)

    # Convert "Date_Time" to datetime format
    print("Convert Date_Time to datetime format :")
    dataset["Date_Time"] = pd.to_datetime(dataset["Date_Time"])
    print(dataset["Date_Time"].head())

    # Extract year, month, day, hour, minute, second from "Date_Time"
    dataset["year"] = dataset["Date_Time"].dt.year
    dataset["month"] = dataset["Date_Time"].dt.month
    dataset["day"] = dataset["Date_Time"].dt.day
    dataset["hour"] = dataset["Date_Time"].dt.hour 
    dataset["minute"] = dataset["Date_Time"].dt.minute 
    dataset["second"] = dataset["Date_Time"].dt.second 
    print("-"*140)

    # Drop the original "Date_Time" column
    print("Dataset after droping Date_Time column :")
    dataset.drop(columns=["Date_Time"],inplace=True)
    print(dataset.head())
    print("-"*140)
    return dataset

def train_and_evaluate_models(dataset):
    # Define features (X) and target (y)
    x = dataset.drop("Wind_Speed_kmh",axis=1)
    y = dataset["Wind_Speed_kmh"]

    # Split the data into training and testing sets (70% training, 30% testing)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

    sc = StandardScaler()
    scaled_x_train = sc.fit_transform(x_train)
    scaled_x_test = sc.transform(x_test)

    # Convert scaled arrays back to DataFrame to keep column names
    scaled_x_train = pd.DataFrame(scaled_x_train, columns=x.columns)
    scaled_x_test = pd.DataFrame(scaled_x_test, columns=x.columns)

    # Initialize and train the Linear Regression model
    reg = LinearRegression()
    reg.fit(scaled_x_train,y_train)

    # Make predictions on the test data
    prediction = reg.predict(scaled_x_test)

    # Calculate Mean Squared Error (MSE) of the predictions
    mse = mean_squared_error(y_test,prediction)
    print(f"Mean Squared Error :{mse:.2f}")
    rmse = sqrt(mse)
    print(f"Root Mean Squared Error :{rmse:.2f}")
    r2 = r2_score(y_test,prediction)
    print(f"r2 score :{r2:.2f}")

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_test, y=prediction, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
    plt.title("Actual vs. Predicted Wind Speed")
    plt.xlabel("Actual Wind Speed")
    plt.ylabel("Predicted Wind Speed")
    plt.show()
    
def main():
    print("Climate Change Prediction")
    dataset = weather()
    dataset = preprocess_data(dataset=dataset)
    train_and_evaluate_models(dataset=dataset)
if __name__=="__main__":
    main()









