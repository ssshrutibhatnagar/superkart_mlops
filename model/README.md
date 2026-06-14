
---
library_name: scikit-learn
pipeline_tag: tabular-regression
---

# SuperKart Sales Forecasting Model

This model predicts `Product_Store_Sales_Total` using product and store attributes.

## Features

- Numeric features: ['Product_Weight', 'Product_Allocated_Area', 'Product_MRP']
- Categorical features: ['Product_Sugar_Content', 'Product_Type', 'Store_Establishment_Year', 'Store_Size', 'Store_Location_City_Type', 'Store_Type']

## Final Selected Model

Random Forest Base

## Model Selection Reason

The Random Forest Base model was selected because it gave the best test performance among the evaluated models, with lower RMSE and higher R-squared on unseen test data compared with the tuned alternatives.

## Intended Use

This model supports sales forecasting for inventory planning, supply-chain procurement, and store-level sales planning.
