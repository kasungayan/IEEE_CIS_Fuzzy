#!/usr/bin/env python
# coding: utf-8

# In[144]:


# Loading the required libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor


# In[226]:


# Random seed for reproducability
np.random.seed(123)


# In[484]:


# Model Training File
df_train = pd.read_feather('df_train.feather')


# In[300]:


# Imputed File
df_imputed = pd.read_csv("weekday_mean_season_imputed.csv")


# In[302]:


# Replacing with the Imputed Energy Consumption
df_imputed = df_imputed[['energy_agg']]
df_train = df_train.drop(columns = ['energy_agg'])
df_train  = pd.concat([df_train, df_imputed], axis = 1)


# In[489]:


# Reading the file relevant to test file
df_test = pd.read_feather('df_test.feather')


# In[491]:


# Setting dependent variable to zero (test set)
df_test['energy_agg'] = 0


# In[492]:


# Reading the estimated temperature file.
df_temp = pd.read_csv('all_temp_boostrap_median.csv')


# In[493]:


# Selecting the required temperature columns.
df_temp = df_temp[['avg_temp', 'max_temp', 'min_temp']]


# In[496]:


# Removing the existing temperature columns from the test dataframe.
df_test = df_test.drop(columns = ['avg_temp', 'max_temp','min_temp'])


# In[497]:


# Replace the temperature columns with the estimated.
df_test = pd.concat([df_test, df_temp], axis = 1)


# In[501]:


# Combining with the training file.
df_test.rename(columns={'ds': 'date_only'}, inplace=True)
df_combined = pd.concat([df_train, df_test], axis = 0)


# In[503]:


# Replace empty fields as NAs
df_combined.fillna(value=np.nan, inplace=True)


# In[505]:


# Fill categorical NA values with Unknowns
df_combined['dwelling_type'] = df_combined['dwelling_type'].cat.add_categories('Unknown')
df_combined['dwelling_type'].fillna('Unknown', inplace =True) 
df_combined['heating_fuel'] = df_combined['heating_fuel'].cat.add_categories('Unknown')
df_combined['heating_fuel'].fillna('Unknown', inplace =True)
df_combined['hot_water_fuel'] = df_combined['hot_water_fuel'].cat.add_categories('Unknown')
df_combined['hot_water_fuel'].fillna('Unknown', inplace =True)
df_combined['boiler_age'] = df_combined['boiler_age'].cat.add_categories('Unknown')
df_combined['boiler_age'].fillna('Unknown', inplace =True) 
df_combined['loft_insulation'] = df_combined['loft_insulation'].cat.add_categories('Unknown')
df_combined['loft_insulation'].fillna('Unknown', inplace =True) 
df_combined['wall_insulation'] = df_combined['wall_insulation'].cat.add_categories('Unknown')
df_combined['wall_insulation'].fillna('Unknown', inplace =True) 
df_combined['heating_temperature'] = df_combined['heating_temperature'].cat.add_categories('Unknown')
df_combined['heating_temperature'].fillna('Unknown', inplace =True) 
df_combined['efficient_lighting_percentage'] = df_combined['efficient_lighting_percentage'].cat.add_categories('Unknown')
df_combined['efficient_lighting_percentage'].fillna('Unknown', inplace =True)


# In[506]:


# Fill numerical NA values with a predefined number.
df_combined['num_occupants'].fillna(-999,inplace=True)
df_combined['num_bedrooms'].fillna(-999,inplace=True)
df_combined['dishwasher'].fillna(-999,inplace=True)
df_combined['freezer'].fillna(-999,inplace=True)
df_combined['fridge_freezer'].fillna(-999,inplace=True)
df_combined['refrigerator'].fillna(-999,inplace=True)
df_combined['tumble_dryer'].fillna(-999,inplace=True)
df_combined['washing_machine'].fillna(-999,inplace=True)
df_combined['game_console'].fillna(-999,inplace=True)
df_combined['laptop'].fillna(-999,inplace=True)
df_combined['pc'].fillna(-999,inplace=True)
df_combined['router'].fillna(-999,inplace=True)
df_combined['set_top_box'].fillna(-999,inplace=True)
df_combined['tablet'].fillna(-999,inplace=True)
df_combined['tv'].fillna(-999,inplace=True)


# In[509]:


# Define categorical features
cat_features = ['dwelling_type','heating_fuel', 'hot_water_fuel', 'boiler_age', 'loft_insulation', 'wall_insulation', 'heating_temperature', 'efficient_lighting_percentage']


# In[512]:


# Categorical Feature Embedding:
lbl = preprocessing.LabelEncoder()
for col in cat_features:
    df_combined[col] = lbl.fit_transform(df_combined[col].astype(str))


# In[514]:


# Temperature Scaling (0-1)
min_max_scaler = preprocessing.MinMaxScaler()
df_combined[["max_temp", "avg_temp", "min_temp"]] = min_max_scaler.fit_transform(df_combined[["max_temp", "avg_temp", "min_temp"]])


# In[516]:


# Drop unnecessary columns.
df_combined = df_combined.drop(columns = ['meter_id', 'date_only'])


# In[519]:


# Encoding month and day-of-the-week as integer variables.
df_combined['month'] = df_combined['month'].astype('int8')
df_combined['day_of_week'] = df_combined['day_of_week'].astype('int8')


# In[522]:


# Splitting the training and testing data.
df_train_only = df_combined.iloc[0:643287,]
df_test_only = df_combined.iloc[643287:1828807,]


# In[527]:


# Remove the dependent column from the test set.
df_test_only = df_test_only.drop(columns=["energy_agg"])


# In[531]:


# Splitting the training data to train and validation.
y = df_train_only['energy_agg']
X = df_train_only.drop(columns=["energy_agg"])


# In[534]:


# Training and Validation Set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=28)


# In[536]:


# Prepare Categorical Variables
def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]


# In[537]:


categorical_features_pos = column_index(X, cat_features)


# In[540]:


# Defining the Catboost Object.
cb_model = CatBoostRegressor(iterations=700,
                             learning_rate=0.04,
                             depth=12,
                             loss_function = 'Quantile:alpha=0.45',
                             eval_metric = 'RMSE',
                             random_seed = 30,
                             bagging_temperature = 0.4,
                             od_type='Iter',
                             metric_period = 75,
                             od_wait=100)


# In[541]:


# Training the Catboost model.
model = cb_model.fit(X_train, y_train,
             eval_set=(X_valid,y_valid),
             cat_features=categorical_features_pos,
             use_best_model=True,
             verbose=True)


# In[543]:


# Fitting the Catboost model to the training set.
model_fit = model.predict(X)


# In[546]:


df_train_select = df_train[['meter_id', 'date_only']]


# In[547]:


df_train_select['energy_agg'] = model_fit


# In[549]:


# Writing the model fitting to CSV (later used by ARIMA model).
df_train_select.to_csv("catboost_model_season_fitting.csv", index=False)


# In[550]:


# Getting the predictions for the testset.
df_test_select = df_test[['meter_id', 'date_only']]
predictions = model.predict(df_test_only)


# In[553]:


predictions[predictions < 0] = 0


# In[556]:


df_test_select['energy_agg'] = predictions


# In[558]:


# Writing the predictions from the Catboost model.
df_test_select.to_csv("catboost_model_season_forecasts.csv", index=False)

