# exporing how to save pickle files
import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.DataFrame()

df['area'] = [1,2,3,4,5]
df['price'] = [100,333,500,600,900]
df.to_csv('homeprices.csv')

df = pd.read_csv("homeprices.csv")
print(df.head())

model = linear_model.LinearRegression()
model.fit(df[['area']],df.price)

print('coef -> ', model.coef_)
print(' intercept -> ', model.intercept_)
#model.predict(6)

import pickle

with open('test_pickle','wb') as f:
    pickle.dump(model,f)

with open('test_picle','rb') as f:
    model = pickle.load(f)
    #model.predict(6)


