import pandas as pd
  
# reading two csv files
data1 = pd.read_csv('data/country_data copy.csv')
data2 = pd.read_csv('data/Vaccine acceptance rate.csv')
  
# using merge function by setting how='inner'
output1 = pd.merge(data1, data2, 
                   on='country', 
                   how='left')
# displaying result
output1.to_csv('data/newcountrydata.csv',index=False)
print(output1)