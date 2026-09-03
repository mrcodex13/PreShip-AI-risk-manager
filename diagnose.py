"""Diagnostic analysis of signal quality."""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

data = pd.read_csv('data/Export_Product_Return_Data.csv')
data = data.drop(columns=[col for col in data.columns if col.startswith('Unnamed:')])
data = data.rename(columns={'high_return_risk': 'High_Return_Risk'})
data['target'] = data['High_Return_Risk'].astype(str).str.lower().eq('yes').astype(int)

numeric_features = ['Age', 'Quantity', 'Price', 'Discount', 'Product Rating']

print('=== CLASS MEANS (NUMERIC FEATURES) ===')
print(data.groupby('target')[numeric_features].mean())

print('\n=== MUTUAL INFORMATION WITH TARGET ===')
mi = mutual_info_classif(data[numeric_features].fillna(data[numeric_features].median()), data['target'], random_state=42)
mi_df = pd.DataFrame({'Feature': numeric_features, 'MI': mi}).sort_values('MI', ascending=False)
print(mi_df)

print('\n=== LABEL BALANCE ===')
print(f'High-risk rate: {data["target"].mean():.2%}')
print(f'Counts: {data["target"].value_counts().to_dict()}')

# Check categorical features
print('\n=== CATEGORICAL FEATURE DISTRIBUTIONS ===')
categorical_features = ['Gender', 'State', 'Category', 'Brand']
for feat in categorical_features:
    print(f'\n{feat}:')
    print(data[feat].value_counts().head())

print('\n=== FEATURE RANGES ===')
print(data[numeric_features].describe())
