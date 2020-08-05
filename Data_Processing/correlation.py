import pandas as pd
import seaborn as sns

dt=pd.read_csv("Resources/Final Database.csv", usecols=["Year","Neighborhood","LAT","LON","Crime Score","Amount"])
dt["Amount"]=dt["Amount"].str.replace(",","").astype(float)

dt2=dt.select_dtypes(exclude=["object","category"])
corr=dt2.corr()

sns.heatmap(dt2.corr())