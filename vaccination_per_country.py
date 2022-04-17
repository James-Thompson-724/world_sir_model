import pandas as pd

# reading two csv files
data1 = pd.read_csv('data/owid-covid-data.csv')
justAprilData = data1['date']=='2022-03-01'
covidData_April2022 = data1[justAprilData]
print(covidData_April2022.head())