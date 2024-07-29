import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Data Cleaning
data = pd.read_csv('housing.csv')
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
imputer = SimpleImputer(strategy='mean')
data['longitude'] = imputer.fit_transform(data[['longitude']])

# Data Transformation
categorical_cols = ['ocean_proximity']
data_transformed = pd.get_dummies(data, columns=categorical_cols, drop_first=True)  # Using pd.get_dummies for one-hot encoding
numerical_cols = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
scaler = StandardScaler()
data_transformed[numerical_cols] = scaler.fit_transform(data_transformed[numerical_cols])

print(data_transformed)
# Save the processed data to a new CSV file
# data_transformed.to_csv('processed_data.csv', index=False)