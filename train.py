
import numpy as np
import pandas as pd
import pickle
from sklearn import preprocessing 
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

df = pd.read_csv('/home/administrator/mlzoomcamp/capstone_project_1/shopping_behavior_updated.csv')
columns_to_drop = ["Age", "Gender", "Size", "Subscription Status", "Discount Applied", "Promo Code Used","Shipping Type", "Color", "Review Rating","Payment Method","Frequency of Purchases"]
df.drop(columns=columns_to_drop, inplace=True)
df.rename(columns={'Purchase Amount (USD)': 'Purchase_Amount_USD'}, inplace=True)

# Encoding all categorical varibale to numeric values

label_encoder = preprocessing.LabelEncoder()
df['Item Purchased']= label_encoder.fit_transform(df['Item Purchased'])
df['Category']= label_encoder.fit_transform(df['Category']) 
df['Location']= label_encoder.fit_transform(df['Location']) 
df['Season']= label_encoder.fit_transform(df['Season']) 

X_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(X_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


y_train = df_train.Purchase_Amount_USD.values
y_val = df_val.Purchase_Amount_USD.values
y_test = df_test.Purchase_Amount_USD.values

del df_train['Purchase_Amount_USD']
del df_val['Purchase_Amount_USD']
del df_test['Purchase_Amount_USD']


def train(df_train, y_train, r=9):
# Create a linear regression model
    model = make_pipeline(StandardScaler(), LinearRegression())

# Train the model
    model.fit(df_train, y_train)

    return model



# Save the model weights to a file
model_filename = 'model.bin'
with open(model_filename, 'wb') as model_file:
    pickle.dump(model_filename, model_file)

# Save the dataframe (dv) to a file
#dv_filename = 'dv.bin'
#with open(dv_filename, 'wb') as dv_file:
   # pickle.dump(dv, dv_file)

