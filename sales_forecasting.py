import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Read the dataset
df = pd.read_csv("dataset/Sales Data.csv")

# Convert 'Order Date' to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Extract additional temporal features from 'Order Date'
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month

# Group by the relevant features and sum sales
product_sales = df.groupby(['City', 'Product', 'Month', 'Year'])['Sales'].sum().reset_index()
print(product_sales.head())

product_sales = pd.get_dummies(product_sales, columns=['City', 'Product'], drop_first=True)

# Features (X) and target (y)
X = product_sales.drop(['Sales'], axis=1)
y = product_sales['Sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")

# Sample data for prediction
data = {
    'City': ['New York City'],
    'Product': ['20in Monitor'],
    'Month': [5],
    'Year': [2027],
}

df_sample = pd.DataFrame(data)

# One-hot encode 'City' and 'Product' in df_sample
df_sample = pd.get_dummies(df_sample, columns=['City', 'Product'], drop_first=True)

# Reindex the sample to match the training columns
df_sample = df_sample.reindex(columns=X.columns, fill_value=0)

# Make prediction using the trained model
y_sample = model.predict(df_sample)
print(f"Predicted sales: {y_sample[0]}")
