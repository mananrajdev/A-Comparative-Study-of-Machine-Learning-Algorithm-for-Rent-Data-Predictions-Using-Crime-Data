import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings

def float_lat_lon(dataset):
    
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    print("Float Lat and Lon - " )
    mae,rmse=regressor(x,y)
    return [mae,rmse]

#--------------------------------------------------------------------------
def float_lat_lon_no_zero_crime(dataset):
    dataset=dataset[dataset["Crime Score"] !=0]
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    print("Float Lat, Lon and no zero crime - " )
    mae,rmse=regressor(x,y)
    return [mae,rmse]

#--------------------------------------------------------------------------
def cat_lat_lon(dataset):
    dataset=dataset[["Year","Crime Score","lat_lon","Amount"]]
    y = dataset.iloc[:, -1].values
    dataset=dataset[["Year","Crime Score","lat_lon"]]
    dataset_onehot= pd.get_dummies(dataset,columns=["lat_lon"])
    
    x = dataset_onehot.values
    
    print("Categorical lat and Lon - ")
    mae,rmse=regressor(x,y)
    return [mae,rmse]

#-----------------------------------------------------------------------------
def cat_lat_lon_year(dataset):
    dataset=dataset[["Year","Crime Score","lat_lon","Amount"]]
    y = dataset.iloc[:, -1].values

    dataset_onehot=dataset[["Year","Crime Score","lat_lon"]].copy()
    dataset_onehot= pd.get_dummies(dataset_onehot,columns=["lat_lon"])
    
    x = dataset_onehot.values
    

    # Encoding categorical data
    # Encoding the Independent Variable
    labelencoder_X = LabelEncoder()
    x[:, 0] = labelencoder_X.fit_transform(x[:, 0])
    onehotencoder = OneHotEncoder(categorical_features = [0])
    x = onehotencoder.fit_transform(x).toarray()
    

    print("Categorical Lat, Lon and year - ")
    mae,rmse=regressor(x,y)
    return [mae,rmse]


#-----------------------------------------------------------------------------
def cat_lat_lon_year_no_zero_crime(dataset):
    dataset=dataset[["Year","Crime Score","lat_lon","Amount"]]
    dataset=dataset[dataset["Crime Score"] !=0]
    y = dataset.iloc[:, -1].values
    
    dataset_onehot=dataset[["Year","Crime Score","lat_lon"]].copy()
    dataset_onehot= pd.get_dummies(dataset_onehot,columns=["lat_lon"])

    
    x = dataset_onehot.values
    

    # Encoding categorical data
    # Encoding the Independent Variable
    labelencoder_X = LabelEncoder()
    x[:, 0] = labelencoder_X.fit_transform(x[:, 0])
    onehotencoder = OneHotEncoder(categorical_features = [0])
    x = onehotencoder.fit_transform(x).toarray()
    

    print("Categorical Lat, Lon, year and no zero crime - ")
    mae,rmse=regressor(x,y)
    return [mae,rmse]

#-----------------------------------------------------------------    
def regressor(x,y):
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
    regressor = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    #regressor = RandomForestRegressor(random_state=0)
    regressor.fit(x_train, y_train)
    
    
    y_pred = regressor.predict(x_test)
    
    
    mae=metrics.mean_absolute_error(y_test,y_pred)
    rmse=np.sqrt(metrics.mean_squared_error(y_test,y_pred))
    print("Mean Absolute Error: ",mae)
    print("Root Mean Squared Error: ",rmse,"\n")
    
    return mae,rmse

#-------------------------------------------------------------------------------

def plot_MAE_RMSE(MAE_RMSE):

    MAE_RMSE_list=MAE_RMSE.values()
    MAE=list(map(lambda x:x[0],MAE_RMSE_list))
    RMSE=list(map(lambda x:x[1],MAE_RMSE_list))
    
    barWidth = 0.25
    # Set position of bar on X axis
    r1 = np.arange(len(MAE_RMSE))
    r2 = [x + barWidth for x in r1]
    
    # Make the plot
    plt.bar(r1, MAE, color='#7f6d5f', width=barWidth, edgecolor='white', label='Mean Absolute Error')
            
         
    plt.bar(r2, RMSE, color='#557f2d', width=barWidth, edgecolor='white', label='Root Mean Squared Error')
     
    # Add xticks on the middle of the group bars
    plt.title('Support Vector Regression')
    plt.xlabel('Errors', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(MAE_RMSE))], MAE_RMSE.keys())
     
    
    
    # Create legend & Show graphic
    plt.legend()
    plt.show()       
    
#-----------------------------------------------------------
if __name__ == "__main__":
    warnings.simplefilter(action='ignore')
    print("Support Vector Regression")
    MAE_RMSE={}
    dt=pd.read_csv("Resources/Final Database.csv", usecols=["Year","LAT","LON","Crime Score","Amount"])
    dt["Amount"]=dt["Amount"].str.replace(",","").astype(float)
    dataset=dt.dropna()
    dataset=dataset.reindex(columns=["Year","LAT","LON","Crime Score","Amount"])
    MAE_RMSE["float_lat_lon"]=float_lat_lon(dataset)    
    MAE_RMSE["float_lat_lon_no_zero_crime"]=float_lat_lon_no_zero_crime(dataset)    
    
#    ----------------------Convert Categorical---------------
    uniq_loc = {}
    lat_long = np.array(dataset[['LAT', 'LON']])
    cat_label = []
    counter = 1
    for i in lat_long:
        val = tuple(i)
        if val not in uniq_loc:
            uniq_loc[val] = 'L'+str(counter)
            counter += 1
        cat_label.append(uniq_loc[val])
    dataset['lat_lon'] = cat_label
    
    MAE_RMSE["cat_lat_lon"]=cat_lat_lon(dataset) 
    MAE_RMSE["cat_lat_lon_year"]=cat_lat_lon_year(dataset)
    MAE_RMSE["cat_lat_lon_year_no_zero_crime"]=cat_lat_lon_year_no_zero_crime(dataset)

    plot_MAE_RMSE(MAE_RMSE)
