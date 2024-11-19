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
df['DayOfWeek'] = df['Order Date'].dt.dayofweek  # 0: Monday, 6: Sunday
df['DayOfYear'] = df['Order Date'].dt.dayofyear

# Use 'Product', 'City', and the new features (Year, Month, DayOfWeek) as features
df = df[['Product', 'City', 'Month', 'Year', 'DayOfWeek', 'Sales']]

# Group by the relevant features
product_sales = df.groupby(['City', 'Product', 'Month', 'Year', 'DayOfWeek'])['Sales'].sum().reset_index()

# One-hot encode 'City' and 'Product'
df = pd.get_dummies(df, columns=['City', 'Product'], drop_first=True)

# Features (X) and target (y)
X = df.drop(['Sales'], axis=1)
y = df['Sales']

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
