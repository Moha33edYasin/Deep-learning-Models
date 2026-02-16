import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate 200 data points for two input features
x1 = np.random.uniform(-10, 10, 200)
x2 = np.random.uniform(-10, 10, 200)

# Define output y as a binary outcome based on a rule
# Example: y = 1 if x1^2 + x2^2 < 10, else 0
y = np.where(x1 + x2 > 0, 1, 0)

# Combine into a pandas DataFrame
data = pd.DataFrame({
    'x1': x1,
    'x2': x2,
    'y': y
})

# Display first 10 rows
print(data.head())

# Optionally, save to CSV
data.to_csv('synthetic_data.csv', index=False)
print("\nDataset saved as 'synthetic_data.csv'")