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
