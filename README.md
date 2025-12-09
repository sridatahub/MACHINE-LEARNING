**Project Overview**

This project focuses on understanding and preparing data for a Machine Learning model that predicts whether a user will purchase a product based on their Age, Estimated Salary, and Gender.

So far, the work includes data loading, exploration, encoding categorical variables, and feature scaling using StandardScaler.

**Dataset Description**
Column	Description
| Column              | Description                                             |
| ------------------- | ------------------------------------------------------- |
| **User ID**         | Unique identifier for each user (not used for modeling) |
| **Gender**          | Gender of the user — Male/Female                        |
| **Age**             | Age of the user in years                                |
| **EstimatedSalary** | Approximate annual salary of the user                   |
| **Purchased**       | Target variable — 1 (Purchased), 0 (Not Purchased)      |

**Steps**
**Data Loading**
```python
# Importing core Python data science libraries
import numpy as np              # For numerical operations
import pandas as pd             # For data handling and analysis
import matplotlib.pyplot as plt # For data visualization (plots and charts)
import seaborn as sns           # For advanced statistical visualizations
'''

```python
## Data loading
df = pd.read_csv("social_media_ads.csv")
df.head()     #Displays first five rows in the dataframe.
df.shape      #(400,5)-->(rows,cols)
```
```python
# Drop the 'User ID' column because it is just an identifier 
# and does not contribute to the model's prediction
df = df.drop('User ID', axis=1)
```
## Encoding Categorical Data

```python
# Convert the 'Gender' column into numeric format
# Male → 1 and Female → 0
df['Gender'] = df['Gender'].replace({'Male': 1, 'Female': 0})
```
##  Splitting Features and Target Variable

```python
# Separate features and target variable
# 'Purchased' is the target column we want to predict
X = df.drop('Purchased', axis=1)
Y = df['Purchased']
```
##  Splitting the Dataset into Training and Testing Sets

```python
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
# test_size=0.3 → 30% data for testing, 70% for training
# random_state=42 → ensures reproducible results
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
```
##  Feature Scaling using Standardization (Z-score Normalization)

**Standardization**, also known as *z-score normalization*, is a commonly used technique in feature scaling.  
It involves **transforming the features in a dataset so that they have a mean of zero and a standard deviation of one**.

```python
from sklearn.preprocessing import StandardScaler

# Create a StandardScaler object
scaler_obj = StandardScaler()

# Fit and transform the training data
X_train_scaled = scaler_obj.fit_transform(x_train)
X_test_scaled = scaler_obj.transform(x_test)
```
##  Converting Scaled Arrays Back to DataFrames

```python
# Convert scaled arrays back into DataFrames for better readability
x_train_df = pd.DataFrame(data=x_train_scaled, columns=x_train.columns)
x_test_df = pd.DataFrame(data=x_test_scaled, columns=x_test.columns)
```
##  Verifying Standardization Results

After applying `StandardScaler`, we can check whether our features have been correctly standardized by using the `describe()` function:

```python
# Check summary statistics after scaling
np.round(x_train_df.describe(), 2)
```

<img width="339" height="319" alt="image" src="https://github.com/user-attachments/assets/68c8e119-9b66-401d-b7a7-cd94488d557a" />

