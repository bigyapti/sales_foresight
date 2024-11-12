import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("dataset/Sales Data.csv")

# Convert Order Date to datetime
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Calculate Year-Month and Total Price
df["Year-Month"] = df["Order Date"].dt.to_period("M")
df["Total Price"] = df["Quantity Ordered"] * df["Price Each"]

# Group data
product_purchase_per_month = df.groupby("Year-Month")["Quantity Ordered"].sum().reset_index()
product_purchase_per_day = df.groupby("Order Date")["Quantity Ordered"].sum().reset_index()
top_ordered_products = df.groupby("Product")["Quantity Ordered"].sum().sort_values(ascending=False)
top_products_by_sales = df.groupby("Product")["Total Price"].sum().sort_values(ascending=False)

# Plot 1: Product Purchase Per Month
plt.figure()
sns.barplot(x="Year-Month", y="Quantity Ordered", data=product_purchase_per_month)
plt.title("Product Purchase Per Month")
plt.xticks(rotation=90)
plt.show()

# Plot 2: Product Purchase Per Day
plt.figure()
sns.barplot(x="Order Date", y="Quantity Ordered", data=product_purchase_per_day)
plt.title("Product Purchase Per Day")
plt.xticks(rotation=45)
plt.show()

# Plot 3: Top Selling Products by Quantity
plt.figure()
sns.barplot(x=top_ordered_products.index, y=top_ordered_products.values)
plt.title("Top Selling Products by Quantity")
plt.xlabel("Product")
plt.ylabel("Quantity Ordered")
plt.xticks(rotation=90)
plt.show()

# Plot 4: Top Selling Products by Total Sales
plt.figure()
sns.barplot(x=top_products_by_sales.index, y=top_products_by_sales.values)
plt.title("Top Selling Products by Total Sales")
plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.xticks(rotation=90)
plt.show()

# Plot 5: Correlation Matrix of Quantity Ordered, Price Each, and Total Price
plt.figure()
correlation_matrix = df[["Quantity Ordered", "Price Each", "Total Price"]].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=1, linecolor='black')
plt.title("Correlation Matrix of Sales Data")
plt.show()
