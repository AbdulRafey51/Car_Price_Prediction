# Car_Price_Prediction
Have you ever wondered how much your car is worth? Imagine being able to predict its price using some clever technology. That’s what we’re going to explore today — predicting car prices using machine learning. 
Don’t worry, I’ll keep it simple and include some code along the way to show you how it all works!

For this project, I used a car sales dataset from Kaggle. I also learned the steps involved from a tutorial by CampusX on YouTube, and I applied several machine learning models to get the best predictions.

Why Predict Car Prices?
Knowing the value of a car is crucial when buying or selling. Machine learning can help estimate car prices based on various features such as:

Year: The year the car was made.
Present Price: The price of the car when it was brand new.
Kms Driven: How many kilometers the car has driven.
Fuel Type: Whether the car runs on petrol, diesel, or CNG.
Using this data, we can train a model that predicts the Selling Price of a car.

The Dataset
The dataset I used has the following key features:

- Year: The year the car was manufactured.
- Present_Price: The current price of the car when new.
- Selling_Price: The price the car is being sold for.
- Kms_Driven: Total kilometers the car has been driven.
- Fuel_Type: The fuel type of the car (Petrol/Diesel/CNG).
Here’s a quick look at the data:

import pandas as pd
car_pr = pd.read_csv('/content/car data.csv')
car_pr.head()
Exploring the Data
Before building models, I visualized the data to better understand it. Visualization helps us spot trends, like how Present_Price and Selling_Price are related.

import plotly.express as px
# Scatter plot of Present Price vs. Selling Price
fig = px.scatter(car_pr, x='Present_Price', y='Selling_Price', color='Fuel_Type', title='Present Price vs. Selling Price')
fig.show()
This graph shows that higher present prices tend to result in higher selling prices, and cars using diesel typically sell for more than those using petrol.

Preparing the Data for Machine Learning
Machine learning models require clean data, so I performed some preprocessing steps. 
I also used One-Hot Encoding to convert categorical features like Fuel_Type into numerical data, which machine learning models can understand.

Step 1: Split the data into features (X) and target (y)
X = car_pr.drop(columns="Selling_Price")
y = car_pr["Selling_Price"]
# Splitting the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
Machine Learning Models
To predict the car price, I tested several machine learning models. Each model has a different way of making predictions. 
Here are a few of the models I used:

1. Linear Regression
Linear Regression draws a straight line that fits the data as best as possible.
It’s like predicting selling prices by calculating how features like Year and Kms_Driven are related to price.

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error
# Pipeline for Linear Regression
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse=False, drop='first', handle_unknown="ignore"), [0, 4, 5, 6])
], remainder='passthrough')
step2 = LinearRegression()
pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print('R2 score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))
2. Decision Tree Regressor
A Decision Tree splits the data based on conditions like 
“Is the car newer than 2015?” or “Is the fuel type petrol?” It creates a tree-like structure to make predictions.

from sklearn.tree import DecisionTreeRegressor
step2 = DecisionTreeRegressor(criterion='squared_error')
pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print('R2 score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))
3. Random Forest Regressor
Random Forest is like a group of decision trees working together. 
It creates multiple trees and averages their predictions to get a more accurate result.

from sklearn.ensemble import RandomForestRegressor
step2 = RandomForestRegressor(n_estimators=100, random_state=3, max_depth=8)
pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print('R2 score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))
4. XGBoost Regressor
XGBoost is one of the most powerful models for prediction tasks. It uses a method called gradient boosting, 
which improves accuracy by focusing on the areas where previous models made mistakes.

from xgboost import XGBRegressor
step2 = XGBRegressor(objective='reg:squarederror')
pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print('R2 score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))
Which Model Worked Best?
I tested several models and compared their performance using two main metrics:

R² (R-Squared): This tells us how well the model fits the data. The closer to 1, the better.
MAE (Mean Absolute Error): This tells us how far off our predictions are from the actual selling prices.
After testing, the Random Forest and XGBoost models performed best, giving the most accurate predictions.

Conclusion
By the end of the project, I was able to predict the price of any car in the dataset with a good level of accuracy. With machine learning, it’s possible to make smart decisions when buying or selling cars, simply by analyzing some key data points.

What’s Next?
In the future, I could improve this model by adding more features, such as the car’s brand or condition. I could also explore more advanced machine learning techniques.
