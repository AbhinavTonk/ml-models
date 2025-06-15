import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Recommendation model

# 1. Sample dataset: Customer purchase behavior
data = {
    'CustomerID': [1, 2, 3, 4, 5],
    'Electronics': [200, 150, 0, 100, 250],
    'Clothing': [0, 50, 100, 0, 30],
    'Groceries': [500, 450, 600, 300, 550],
    'Books': [10, 5, 0, 20, 15]
}

df = pd.DataFrame(data)
df.set_index('CustomerID', inplace=True)

# 2. Take new customer input
new_customer_input = {
    'Electronics': 180,
    'Clothing': 10,
    'Groceries': 400,
    'Books': 5
}

# Convert to DataFrame
new_customer = pd.DataFrame([new_customer_input])

# 3. Train k-NN model on existing customers
knn = NearestNeighbors(n_neighbors=2, metric='euclidean')
knn.fit(df)

# 4. Find similar customers
distances, indices = knn.kneighbors(new_customer)
similar_customers = df.iloc[indices[0]]
print("👥 Similar Customers:\n", similar_customers)

# 5. Recommend categories where similar customers spent more
recommendations = (similar_customers.mean() - new_customer.iloc[0]).sort_values(ascending=False)
recommended_categories = recommendations[recommendations > 0]

print("\n🎯 Recommended Product Categories:")
print(recommended_categories)
