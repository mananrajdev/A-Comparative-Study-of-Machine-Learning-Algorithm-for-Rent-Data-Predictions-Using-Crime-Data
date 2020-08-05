import pandas as pd
import numpy as np
# read dataframe
project_data = pd.read_csv("Resources/Project Database.csv")

# read crime weights
weights = pd.read_csv("Resources/metadata.csv")
crime_weights = np.array(weights["Normalize"])
crime_weights = crime_weights[0:141]
print(crime_weights)

# add crime_score column
project_data["Crime Score"] = 0.0
project_data.drop(columns=["SEX OFFENDER REGISTRANT OUT OF COMPLIANCE"])


for index, rows in project_data.iterrows():
    crime_freq = np.array(rows[6:147])
    project_data.at[index, "Crime Score"] = float(np.dot(crime_freq, crime_weights))

project_data.to_csv("Resources/Final Database.csv")
