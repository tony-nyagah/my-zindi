"""
The objective of this hackathon is to develop a machine learning model to predict what a tourist will spend when visiting Tanzania.    
The model can be used by different tour operators and the Tanzania Tourism Board to automatically help tourists across the world estimate their expenditure before visiting Tanzania.
"""

# %%
import pandas as pd

# %%
train_data = pd.read_csv("data/Train.csv")
test_data = pd.read_csv("data/Test.csv")

X = train_data.drop("total_cost", axis=1)
y = train_data["total_cost"]

X
# %%
