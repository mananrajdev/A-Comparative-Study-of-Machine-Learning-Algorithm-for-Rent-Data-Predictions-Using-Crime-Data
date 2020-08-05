import pandas as pd
import os

LA_Rent_Data = pd.read_csv("Rent_Price__LA_.csv", usecols=["Year", "Amount", "Location", "Neighborhood"], skipinitialspace=True)

LA_Rent_Data[['LAT','LON']] = LA_Rent_Data.Location.str.split(", ",expand=True,)

LA_Rent_Data['LAT'] = LA_Rent_Data['LAT'].str.lstrip('(').astype(float)
LA_Rent_Data['LON'] = LA_Rent_Data['LON'].str.rstrip(')').astype(float)

LA_Rent_Data = LA_Rent_Data.drop("Location",axis=1)

year_filter = LA_Rent_Data.Year == 2010

LA_Rent_Data_sub = LA_Rent_Data[year_filter]

print(LA_Rent_Data.shape)
print(LA_Rent_Data_sub.shape)
print(LA_Rent_Data_sub)

LA_Crime_Data = pd.read_csv("Crime_Data_from_2010_to_2019.csv", usecols=["DATE OCC", "Crm Cd", "Crm Cd Desc",
                                                                         "LAT", "LON"],
                            skipinitialspace=True, )
LA_Crime_Data["Year"] = LA_Crime_Data["DATE OCC"].str[6:10]

crime_names = LA_Crime_Data["Crm Cd Desc"].unique()
sorted_crime_names = sorted(crime_names)

for crime in sorted_crime_names:
    crime = str(crime)
    crime = crime.strip('"')
    print(crime)
    LA_Rent_Data[crime] = 0

for year in range(2010, 2020):
    year_filter = LA_Rent_Data.Year == year

    LA_Rent_Data_sub = LA_Rent_Data[year_filter]

    year_filter = LA_Crime_Data.Year == str(year)
    LA_Crime_Data_sub = LA_Crime_Data[year_filter]

    for index, rows in LA_Rent_Data_sub.iterrows():
        curr_lat = rows["LAT"]
        curr_lon = rows["LON"]

        east_filter = curr_lon + 0.005 > LA_Crime_Data_sub.LON
        west_filter = LA_Crime_Data_sub.LON > curr_lon - 0.005
        north_filter = curr_lat + 0.005 > LA_Crime_Data_sub.LAT
        south_filter = LA_Crime_Data_sub.LAT > curr_lat - 0.005

        location_filters = east_filter & west_filter & north_filter & south_filter

        NEARBY_CRIME = LA_Crime_Data_sub[location_filters]

        for index2, drows in NEARBY_CRIME.iterrows():

            crime_name = drows["Crm Cd Desc"]
            LA_Rent_Data_sub.at[index, crime_name] += 1

    if year == 2010:
        LA_Rent_Data_sub.to_csv("Resources/Project Database.csv")
    else:
        LA_Rent_Data_sub.to_csv("Resources/Project Database.csv", mode='a', header=False)

