# Polynomial Regression Model

## Overview
This project demonstrates the development and analysis of a Polynomial Regression model. The model is built using a dataset to explore the relationship between dependent and independent variables, enhancing predictive accuracy through polynomial features.

## Concepts Covered
1. **Importing Libraries**: Importing necessary Python libraries to work with data and build models.
2. **Reading Dataset**: Loading and exploring the dataset to understand the underlying data structure.
3. **Training Linear Regression Model**: Building and training a simple linear regression model as a baseline.
4. **Training Polynomial Regression Model**: Extending the linear model to polynomial regression to capture non-linear relationships.
5. **Visualizing Linear Regression Model**: Plotting the linear regression model to compare the fit with actual data.
6. **Visualizing Polynomial Regression Model**: Visualizing the polynomial regression model to see how it better fits the data compared to the linear model.

## Getting Started
To run this project on your local machine, follow these steps:

### Prerequisites
- Python 3.x
- Required Libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`

### Installation
```bash
pip install numpy pandas matplotlib scikit-learn
```

### Running the Project
1. **Import Libraries**:
   ```python
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   from sklearn.linear_model import LinearRegression
   from sklearn.preprocessing import PolynomialFeatures
   ```

2. **Reading Dataset**:
   ```python
   dataset = pd.read_csv('your_dataset.csv')
   X = dataset.iloc[:, 1:-1].values
   y = dataset.iloc[:, -1].values
   ```

3. **Training Linear Regression Model**:
   ```python
   lin_reg = LinearRegression()
   lin_reg.fit(X, y)
   ```

4. **Training Polynomial Regression Model**:
   ```python
   poly_reg = PolynomialFeatures(degree=4)
   X_poly = poly_reg.fit_transform(X)
   poly_lin_reg = LinearRegression()
   poly_lin_reg.fit(X_poly, y)
   ```

5. **Visualizing Linear Regression Model**:
   ```python
   plt.scatter(X, y, color='red')
   plt.plot(X, lin_reg.predict(X), color='blue')
   plt.title('Linear Regression')
   plt.show()
   ```

6. **Visualizing Polynomial Regression Model**:
   ```python
   plt.scatter(X, y, color='red')
   plt.plot(X, poly_lin_reg.predict(poly_reg.fit_transform(X)), color='blue')
   plt.title('Polynomial Regression')
   plt.show()
   ```

## Conclusion
This project showcases how polynomial regression can be used to model complex relationships between variables, improving upon the limitations of simple linear regression.
