import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


def fit_model(x,y):
    regressor_mlp = MLPRegressor(hidden_layer_sizes=(150, 50,), solver='adam', activation='relu',
                                 learning_rate_init=0.1, max_iter=1000)
    regressor_mlp.fit(x, y)
    return regressor_mlp


def predict(regressor, x, y):
    pred_value = regressor.predict(x)
    return pred_value


def run_regressor(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    regressor = fit_model(x_train, y_train)
    pred_y = predict(regressor, x_test , y_test)
    mae = metrics.mean_absolute_error(y_test, pred_y)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, pred_y))
    print("Mean Absolute Error- ", mae)
    print("Root Mean Squared Error- ", rmse, "\n")
    return [mae, rmse]


def float_lat_lon(dataset):
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    print("Float Lat and Lon - ")
    return run_regressor(x,y)


def float_lat_lon_zero_crime(dataset):
    dataset = dataset[dataset["Crime Score"] != 0]
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    print("Float Lat and Lon - ")
    return run_regressor(x,y)


def cat_lat_lon(dataset):
    dataset = dataset[["Year", "Crime Score", "lat_lon", "Amount"]]
    y = dataset.iloc[:, -1].values
    dataset = dataset[["Year", "Crime Score", "lat_lon"]]
    dataset_onehot = pd.get_dummies(dataset, columns=["lat_lon"])
    x = dataset_onehot.values
    return run_regressor(x,y)


def cat_lat_lon_zero_crime(dataset):
    dataset = dataset[["Year", "Crime Score", "lat_lon", "Amount"]]
    dataset = dataset[dataset["Crime Score"] != 0]
    y = dataset.iloc[:, -1].values
    dataset = dataset[["Year", "Crime Score", "lat_lon"]]
    dataset_onehot = pd.get_dummies(dataset, columns=["lat_lon"])
    x = dataset_onehot.values
    return run_regressor(x,y)


def cat_lat_lon_year(dataset):
    dataset = dataset[["Year", "Crime Score", "lat_lon", "Amount"]]
    y = dataset.iloc[:, -1].values
    dataset_onehot = dataset[["Year", "Crime Score", "lat_lon"]].copy()
    dataset_onehot = pd.get_dummies(dataset_onehot, columns=["lat_lon"])
    x = dataset_onehot.values
    labelencoder_X = LabelEncoder()
    x[:, 0] = labelencoder_X.fit_transform(x[:, 0])
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
    x = np.array(columnTransformer.fit_transform(x), dtype=np.float)
    return run_regressor(x,y)


def cat_lat_lon_year_zero_crime(dataset):
    dataset=dataset[["Year","Crime Score","lat_lon","Amount"]]
    dataset=dataset[dataset["Crime Score"] !=0]
    y = dataset.iloc[:, -1].values
    dataset_onehot = dataset[["Year", "Crime Score", "lat_lon"]].copy()
    dataset_onehot = pd.get_dummies(dataset_onehot, columns=["lat_lon"])
    x = dataset_onehot.values
    labelencoder_X = LabelEncoder()
    x[:, 0] = labelencoder_X.fit_transform(x[:, 0])
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
    x = np.array(columnTransformer.fit_transform(x), dtype=np.float)
    return run_regressor(x, y)


def plot_MAE_RMSE(MAE_RMSE):
    MAE_RMSE_list = MAE_RMSE.values()
    MAE = list(map(lambda x: x[0], MAE_RMSE_list))
    RMSE = list(map(lambda x: x[1], MAE_RMSE_list))
    barWidth = 0.25
    # Set position of bar on X axis
    r1 = np.arange(len(MAE_RMSE))
    r2 = [x + barWidth for x in r1]

    # Make the plot
    plt.bar(r1, MAE, color='#7f6d5f', width=barWidth, edgecolor='white', label='Mean Absolute Error')

    plt.bar(r2, RMSE, color='#557f2d', width=barWidth, edgecolor='white', label='Root Mean Squared Error')

    # Add xticks on the middle of the group bars
    plt.xlabel('Errors', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(MAE_RMSE))], MAE_RMSE.keys())

    # Create legend & Show graphic
    plt.legend()
    plt.show()


if __name__ == "__main__":
    MAE_RMSE = {}
    dt = pd.read_csv("Resources/Final Database.csv", usecols=["Year", "LAT", "LON", "Crime Score", "Amount"])
    dt["Amount"] = dt["Amount"].str.replace(",", "").astype(float)
    dataset = dt.dropna()
    dataset = dataset.reindex(columns=["Year", "LAT", "LON", "Crime Score", "Amount"])
    print("Float Lat - Long:")
    MAE_RMSE["Input1"] = float_lat_lon(dataset)
    print("Float Lat-Long No Zero Crime:")
    MAE_RMSE["Input2"] = float_lat_lon_zero_crime(dataset)
    #    ----------------------Convert Categorical---------------
    uniq_loc = {}
    lat_long = np.array(dataset[['LAT', 'LON']])
    cat_label = []
    counter = 1
    for i in lat_long:
        val = tuple(i)
        if val not in uniq_loc:
            uniq_loc[val] = 'L' + str(counter)
            counter += 1
        cat_label.append(uniq_loc[val])
    dataset['lat_lon'] = cat_label
    print("Categorical Lat - Long:")
    MAE_RMSE["Input3"] = cat_lat_lon(dataset)
    print("Categorical Lat - Long - Year:")
    MAE_RMSE["Input4"]=cat_lat_lon_year(dataset)
    print("Categorical Lat - Long - Year no zero crime:")
    MAE_RMSE["Input5"]=cat_lat_lon_year_zero_crime(dataset)
    plot_MAE_RMSE(MAE_RMSE)
    print("end")
