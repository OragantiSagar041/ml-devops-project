
from sklearn.datasets import load_iris
import pandas as pd
from pathlib import Path

# Load the Iris dataset
iris = load_iris(as_frame=True)

# Get the flower measurements
data = iris.data

# Add the flower species column
data["species"] = iris.target.map(
    dict(enumerate(iris.target_names))
)

# Create data folder
Path("data").mkdir(exist_ok=True)

# Save dataset as CSV
data.to_csv("data/iris.csv", index=False)

print("Iris dataset created successfully!")
print(data.head())
