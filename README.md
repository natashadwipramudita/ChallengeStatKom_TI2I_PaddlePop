---
# Challenge Computational Statistics

__Member of Paddle Pop Group__
- Natasha Dwi Pramudita&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (2141720147)
- Raynor Herfian Iqbal Fawwaz&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (2141720260)
- Rheno Rayhan Fayyaz Dhana Pramudya&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (2141720157)
- Saefulloh Fatah Putra Kyranna &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (2141720067)
- Versacitta Feodora Ramadhani&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (2141720156)

---
## "Predictive Modeling for Anticipating Medication 'Wearing-Off' in Parkinson's Disease: Optimizing Treatment Strategies and Symptom Management"
> _This task serves as a comprehensive endeavor undertaken by Mr. Septian Enggar Sukmana, S.Pd., M.T., as part of his final examination assignment for the Computational Statistics class. It focuses on the development of a predictive modeling approach to anticipate medication 'wearing-off' in Parkinson's Disease, with the aim of optimizing treatment strategies and enhancing symptom management._
>> __*Background*__ - _Over time, the medicine’s effective time shortens, causing discomfort among PD patients. Thus, PD patients and clinicians must monitor and record the patient symptom changes for adequate treatment. Parkinson's disease (PD) is a slowly progressive nervous system condition caused by the loss of dopamine-producing brain cells. It primarily affects the patient's motor abilities, but it also has an impact on non-motor functions over time. Patients' symptoms include tremors, muscle stiffness, and difficulty walking and balancing. Then it disrupts the patients' sleep, speech, and mental functions, affecting their quality of life (QoL). With the improvement of wearable devices, patients can be continuously monitored with the help of cell phones and fitness trackers. We use these technologies to monitor patients' health, record WO periods, and document the impact of treatment on their symptoms in order to predict the wearing-off phenomenon in Parkinson's disease patients._

To calculate and configure how the distribution of the patients, graph, distribution score, and prediction line will be needed. Prediction created based on the heart rate, stress score, and also last drug was taken. Prediction used to help the doctors to create specific treatment strategies to manage the Parkinson’s disease and its associated symptoms properly. Graph one of the common method to visually illustrate relationships in the data. The purpose of graph itself is to represent data that are too numerous and random or complicated, so it can be described adequately in the text and take less space.


### Work Documentation
This document provides a detailed description of the work and analysis conducted. It includes code snippets from the data.

#### Importing Libraries
```python
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
```
- `import pandas as pd` imports the pandas library, which is commonly used for data manipulation and analysis in Python. It is often imported under the alias `pd` for convenience.
- `from sklearn import linear_model` imports the `linear_model` module from the scikit-learn library (sklearn). Scikit-learn is a popular machine learning library in Python, and the `linear_model` module provides various linear regression models and related functionalities.
- `import matplotlib.pyplot as plt` imports the pyplot module from the matplotlib library, which is a plotting library in Python. It is commonly imported under the alias `plt` for easier usage.
- `import seaborn as sns` imports the seaborn library, which is another popular data visualization library in Python. Seaborn provides a high-level interface for creating attractive statistical graphics.
- `import numpy as np` imports the numpy library, which is a fundamental package for scientific computing in Python. It provides support for large, multi-dimensional arrays and various mathematical operations.
- `import statsmodels.api as sm` imports the statsmodels library, which is a powerful library for statistical modeling and analysis in Python. The `api` module within statsmodels provides an extensive set of statistical models and functions.

#### Importing the Dataset
```python
df = pd.read_csv(r'C:\Users\Lenovo\Dropbox\PC\Documents\College Assignments\4th Semester\Computation Statistics\Final\datanew.csv')
```
- `pd` refers to the pandas library that imported earlier.
- `read_csv()` is a pandas function used to read data from a CSV file into a DataFrame.
- The argument passed to `read_csv()` is the file path of the CSV file that want to import. In this case, the file is located at `'C:\Users\Lenovo\Dropbox\PC\Documents\College Assignments\4th Semester\Computation Statistics\Final\datanew.csv'`. The `r` prefix before the file path is used to interpret the string as a raw string, which ensures that special characters in the file path are not escaped.
- The resulting DataFrame is assigned to the variable `df`, which can be used to access and manipulate the data.
- _Make sure to replace the file path with the actual path to the `datanew.csv` file._

#### Heart Rate VS Wear Off Duration Scatter Plot
##### This code creates a scatter plot to visualize the relationship between the "Heart Rate" and "Wear Off Duration" variables.
```python
plt.scatter(df['heart_rate'], df['wo_duration'], color='red')
plt.title('Heart Rate VS Wear Off Duration', fontsize=14)
plt.xlabel('Heart Rate', fontsize=14)
plt.ylabel('Wear Off Duration', fontsize=14)
plt.grid(True)
plt.savefig("HeartRate.jpg")
plt.show()
```
- `plt.scatter(df['heart_rate'], df['wo_duration'], color='red')`, creates a scatter plot using `plt.scatter()` from the `matplotlib.pyplot` module. It plots the "heart_rate" values from the DataFrame `df` on the x-axis and the "wo_duration" values on the y-axis. The points in the scatter plot will be displayed in red.
- `plt.title('Heart Rate VS Wear Off Duration', fontsize=14)`
  `plt.xlabel('Heart Rate', fontsize=14)`
  `plt.ylabel('Wear Off Duration', fontsize=14)`, these lines set the title, x-axis label, and y-axis label for the scatter plot. The fontsize=14 argument sets the font size of the title and axis labels to 14.
- `plt.grid(True)`, adds a grid to the scatter plot, making it easier to interpret the data points.
- `plt.savefig("HeartRate.jpg")`, saves the scatter plot as an image file named "HeartRate.jpg" in the current directory.
- `plt.show()`, displays the scatter plot on the screen.

#### Stress Score VS Wear Off Duration Scatter Plot
##### This code creates a scatter plot to visualize the relationship between the "Stress Score" and "Wear Off Duration" variables.
```python
plt.scatter(df['stress_score'], df['wo_duration'], color='red')
plt.title('Stress Score VS Wear Off Duration', fontsize=14)
plt.xlabel('Stress Score', fontsize=14)
plt.ylabel('Wear Off Duration', fontsize=14)
plt.grid(True)
plt.savefig("StressScore.jpg")
plt.show()
```
- `plt.scatter(df['stress_score'], df['wo_duration'], color='red')`, creates a scatter plot using `plt.scatter()` from the `matplotlib.pyplot` module. It plots the "stress_score" values from the DataFrame `df` on the x-axis and the "wo_duration" values on the y-axis. The points in the scatter plot will be displayed in red.
- `plt.title('Stress Score VS Wear Off Duration', fontsize=14)`
  `plt.xlabel('Stress Score', fontsize=14)`
   plt.ylabel('Wear Off Duration', fontsize=14)`, these lines set the title, x-axis label, and y-axis label for the scatter plot. The fontsize=14 argument sets the font size of the title and axis labels to 14.
- `plt.grid(True)`, adds a grid to the scatter plot, making it easier to interpret the data points.
- `plt.savefig("StressScore.jpg")`, saves the scatter plot as an image file named "StressScore.jpg" in the current directory.
- `plt.show()`, displays the scatter plot on the screen.

#### Time From Last Drug Taken VS Wear Off Duration Scatter Plot
##### This code creates a scatter plot to visualize the relationship between the "Time From Last Drug Taken" and "Wear Off Duration" variables.
```python
plt.scatter(df['time_from_last_drug_taken'], df['wo_duration'], color='red')
plt.title('Time From Last Drug Taken VS Wear Off Duration', fontsize=14)
plt.xlabel('Time From Last Drug Taken', fontsize=14)
plt.ylabel('Wear Off Duration', fontsize=14)
plt.grid(True)
plt.savefig("DrugTime.jpg")
plt.show()
```
- `plt.scatter(df['time_from_last_drug_taken'], df['wo_duration'], color='red')`, creates a scatter plot using `plt.scatter()` from the `matplotlib.pyplot` module. It plots the "time_from_last_drug_taken" values from the DataFrame `df` on the x-axis and the "wo_duration" values on the y-axis. The points in the scatter plot will be displayed in red.
- `plt.title('Time From Last Drug Taken VS Wear Off Duration', fontsize=14)`
  `plt.xlabel('Time From Last Drug Taken', fontsize=14)`
  `plt.ylabel('Wear Off Duration', fontsize=14)`, these lines set the title, x-axis label, and y-axis label for the scatter plot. The fontsize=14 argument sets the font size of the title and axis labels to 14.
- `plt.grid(True)`, adds a grid to the scatter plot, making it easier to interpret the data points.
- `plt.savefig("DrugTime.jpg")`, saves the scatter plot as an image file named "DrugTime.jpg" in the current directory.
- `plt.show()`, displays the scatter plot on the screen.

#### Set IV and DV
##### This code sets the independent variable (IV) and dependent variable (DV) for the analysis.
```python
x = df[['heart_rate', 'stress_score', 'time_from_last_drug_taken']]
y = df['wo_duration']
```
- `x = df[['heart_rate', 'stress_score', 'time_from_last_drug_taken']]`, assigns the DataFrame `df` columns `'heart_rate'`, `'stress_score'`, and `'time_from_last_drug_taken'` to the variable `x`. The double brackets `[['...']]` are used to select multiple columns as a DataFrame rather than a Series.
- `y = df['wo_duration']`, assigns the column `'wo_duration'` from the DataFrame `df` to the variable `y`. Here, `y` represents the dependent variable or the variable that are trying to predict or explain.

#### Calculate Regression with sklearn
##### This code performs linear regression using scikit-learn (sklearn) library.
```python
regr = linear_model.LinearRegression()
regr.fit(x, y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
```
- `regr = linear_model.LinearRegression()`, creates an instance of the LinearRegression class from the linear_model module in scikit-learn and assigns it to the variable `regr`. This instance represents the linear regression model.
- `regr.fit(x, y)`, fits the linear regression model to the data. It uses the `fit()` method of the `regr` object, where `x` represents the independent variable(s) and `y` represents the dependent variable. The model learns the relationship between `x` and `y` and calculates the coefficients and intercept.
- `print('Intercept: \n', regr.intercept_)`
  `print('Coefficients: \n', regr.coef_)`, these lines print the intercept and coefficients of the linear regression model. The `intercept_` attribute of the `regr` object represents the y-intercept of the regression line, and the `coef_` attribute represents the coefficients of the independent variables.
- _Make sure already imported the necessary libraries (such as `pandas`, `sklearn.linear_model`) before using this code._

#### Calculate Regression with statsmodel
##### This code performs linear regression using the statsmodels library.
```python
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
predictions = model.predict(x)

print_model = model.summary()
print(print_model)
```
- `x = sm.add_constant(x)`, adds a constant term to the independent variable(s) `x`. The `sm.add_constant()` function from the statsmodels.api module is used to add a column of ones to the DataFrame `x`, which is necessary for calculating the intercept in the linear regression model.
- `model = sm.OLS(y, x).fit()`, fits an ordinary least squares (OLS) model to the data. The `sm.OLS()` function from the statsmodels.api module is used to specify the dependent variable `y` and the independent variables `x`. The `fit()` method is then called on the OLS model to estimate the coefficients and fit the model to the data.
- `predictions = model.predict(x)`, fitted model to generate predictions for the dependent variable using the independent variables `x`. The `predict()` method of the `model` object is called to obtain the predicted values.
- `print_model = model.summary()`
  `print(print_model)`, these lines generate a summary of the linear regression model and print it. The `summary()` method of the `model` object provides a comprehensive summary of the regression results, including the coefficients, standard errors, p-values, R-squared, and other statistics.
- _Make sure you have imported the necessary libraries (such as `pandas`, `statsmodels.api`) before using this code._

#### Residual Plot Line Model
#### This code 
```python
from sklearn.linear_model import LinearRegression
x = df['heart_rate']
```
- import the LinearRegression class from the `sklearn.linear_model` module and assign the 'heart_rate' column from the DataFrame `df` to the variable `x`. This column will be used as the independent variable.
```python
model = LinearRegression()
```
```python
model.fit(x.values.reshape(-1, 1), y)
```
- These lines create an instance of the `LinearRegression` class and fit the model to the data. The `fit()` method is called on the `model` object, where `x` represents the independent variable and `y` represents the dependent variable. The model learns the relationship between `x` and `y` and estimates the coefficients.
```python
model.intercept_
```
```python
model.coef_
```
- These lines retrieve the intercept and coefficients of the linear regression model. The `intercept_` attribute represents the y-intercept of the regression line, and the `coef_` attribute represents the coefficient(s) of the independent variable(s).
```python
# Plots the data points and the regression
fig = plt.figure(figsize = (10,5))
ax = plt.axes()
sns.scatterplot(x = x, y = y, ax = ax)
sns.lineplot(x = [0, 10], y = [model.intercept_, (10 * model.coef_[0] + model.intercept_)], ax = ax, color = 'b')

# Plots the residuals
for i, j in zip(x, y):
    yreg = i * model.coef_[0] + model.intercept_
    ax.plot([i, i], [j, yreg], color = 'r')

plt.savefig('HeartRateLine.jpg')
```
- These lines create the residual plot. The `fig` and `ax` objects are created to set the figure size and axes for the plot. The scatter plot of the data points is created using `sns.scatterplot()`, and the regression line is plotted using `sns.lineplot()`. The line is defined by two points: (0, intercept) and (10, 10 * coefficient + intercept). The residuals are plotted as vertical lines between the data points and the corresponding predicted values on the regression line. The plot is saved as an image file named 'HeartRateLine.jpg'.
- _Make sure already imported the necessary libraries (such as `sklearn.linear_model`, `matplotlib.pyplot`, `seaborn`) before using this code._

---
So this is how we analyze the "Wearing-Off" periods of Parkinson's Disease patients. It's unfortunare that Parkinson's Disease is a relatively common neurogenerative disorder. It disrupted the quality of life of many people worldwide. It's good to grow more on the Parkinson's Disease patient, since we can see many patterns of medicine's wear-off periods. If this data can be used more, it is certain that it would benefit the clinicians around the world.

---
© 2023 Paddle Pop Group - TI 2I | All rights reserved.

