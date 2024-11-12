import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
 

df = pd.read_csv("dataset/Sales Data.csv")

df["Order Date"] = pd.to_datetime(df["Order Date"])

df["Year-Month"] = df["Order Date"].dt.to_period("M")

df["Total Price"] = df["Quantity Ordered"] * df["Price Each"]

product_total_price = df.groupby("Product")["Total Price"].sum()

product_purchase_per_day = df.groupby("Order Date")["Quantity Ordered"].sum()

product_purchase_per_month = df.groupby("Year-Month")["Quantity Ordered"].sum()

top_products_by_sales = df.groupby("Product")["Total Price"].sum().sort_values(ascending=False)

top_ordered_products = df.groupby("Product")["Quantity Ordered"].sum().sort_values(ascending=False)

product_purchase_per_month = product_purchase_per_month.reset_index()  # Convert to DataFrame
sns.lineplot(x="Year-Month", y="Quantity Ordered", data=product_purchase_per_month)
plt.title("Product Purchase Per Month")
plt.show()

print(product_total_price)
print(product_purchase_per_day)
print(product_purchase_per_month)
print(top_ordered_products)
print(top_products_by_sales)
print(df.head())  # To see the first few rows of the dataframe
