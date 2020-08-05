Following are list of files used for this project:

Resources: Folder containg the CSV files used
Project Database: Data contained merged data from both Rent and Crime Datasets. 
metadata.csv: Data containing the crime scores.
Final Database.csv: Data used to conduct ML Algorithms.


Data Processing: Folder containg Python Scripts for various Data Pre-Processing
DataProcessingFinal.py: Contains the code for merging the two datasets: Rent Dataset and Crime Data set. 
In order to run this file, 
1) Download:
Crime Dataset from https://data.lacity.org/A-Safe-City/Crime-Data-from-2010-to-2019/63jg-8b9z/data, and
Rent Dataset from https://usc.data.socrata.com/Los-Angeles/Rent-Price-LA-/4a97-v5tx.
2) Save the files in the same folder. 

AddCrimeScore.py: Contains the code for adding the crime scores using the 
 


Algorithm Implementation: Folder containing Python Scripts for various ML Algorithms
linear_regression.py: Contains the linear regression model training and testing for rent prediction.
randomforests.py: Contains the random forests model training and testing for rent prediction.
svr.py: Contains the support vector regression model training and testing for rent prediction.
nn.py: Contains the neural network model training and testing for rent prediction. 

Scikit-Learn versions:
linear_regression.py, randomforests.py, svr.py: 0.21.3
nn.py: 0.22.3


