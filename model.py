import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import pickle
import datetime
from sklearn.feature_selection import SelectKBest, f_regression 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

chiller_data = pd.read_csv('./dataset/PLANT TON_EFFICIENCY_TEMPERATURE/PLANT TON_EFFICIENCY/TableData (6).csv')
ambient_data = pd.read_excel('./dataset/PLANT TON_EFFICIENCY_TEMPERATURE/TEMPERATURE/ECCO 19400(19400) sensors data (4).xlsx')

chiller_data['Time'] = pd.to_datetime(chiller_data['Time'])
ambient_data['DateTime'] = pd.to_datetime(ambient_data['DateTime'])
chiller_data['Time'] = chiller_data['Time'].dt.round('10min')
ambient_data['DateTime'] = ambient_data['DateTime'].dt.round('10min')
ambient_data.rename(columns={'DateTime': 'Time'}, inplace=True)
dataset = pd.merge(chiller_data, ambient_data, on='Time', how='inner')

def classify_weekday_or_weekend(date):
    try:
        if isinstance(date, str):
            date = datetime.datetime.strptime(date, '%Y-%m-%d')
        elif isinstance(date, pd.Timestamp):
            date = date.to_pydatetime()
        
        day_of_week = date.weekday() 

        return 1 if day_of_week >= 5 else 0
    except Exception as e:
        print(f"Error in date conversion: {e}")
        return -1 

dataset['Time'] = pd.to_datetime(dataset['Time'])

dataset['Day_Type'] = dataset['Time'].apply(lambda x: classify_weekday_or_weekend(x))

data = {
    'Time': dataset['Time']
}

df = pd.DataFrame(data)

df['Time'] = pd.to_datetime(df['Time'])

df['Time_Float'] = df['Time'].astype(int) / 10**9 

dataset['Time'] = df['Time_Float']

X = pd.concat([dataset.iloc[:, :4], dataset.iloc[:, 5:]], axis=1)
y = dataset.iloc[:, 4].values

if isinstance(y, np.ndarray) and y.ndim > 1:
    y = y.flatten() 

if not isinstance(X, pd.DataFrame):
    raise TypeError("X should be a Pandas DataFrame after conversion.")

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)  

X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

feature_selector = SelectKBest(f_regression, k="all")  
fit = feature_selector.fit(X_scaled, y)

p_values = pd.DataFrame(fit.pvalues_, columns=['p_value'])
scores = pd.DataFrame(fit.scores_, columns=['f_score'])
input_variable_names = pd.DataFrame(X.columns, columns=['input_variable'])

summary_stats = pd.concat([input_variable_names, p_values, scores], axis=1)
summary_stats.sort_values(by="p_value", inplace=True)

p_value_threshold = 0
score_threshold = 50

selected_variables = summary_stats.loc[
    (summary_stats["f_score"] >= score_threshold) &
    (summary_stats["p_value"] <= p_value_threshold),
    "input_variable"
].tolist()
selected_variables.sort()

X = X[selected_variables]

regressor = LinearRegression()
regressor.fit(X, y)

pickle.dump(regressor, open('../model/ml_model.pkl', 'wb'))
