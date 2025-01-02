# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1. Start the program.
 2. Data preprocessing:
 3. Cleanse data,handle missing values,encode categorical variables.
 4. Model Training:Fit logistic regression model on preprocessed data.
 5. Model Evaluation:Assess model performance using metrics like accuracyprecisioon,recall.
 6. Prediction: Predict placement status for new student data using trained model.
 7. End the program.

## Program:
```python
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:  Abdul Rasak . N
RegisterNumber:  24002896
*/
# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv("datasets/Placement_Data.csv")
print(data.head())  # To verify the data is loaded correctly

# Create a copy and drop irrelevant columns
data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis=1)
print(data1.head())

# Check for missing and duplicate values
print("Missing values per column:\n", data1.isnull().sum())
print("Number of duplicate rows:", data1.duplicated().sum())

# Encode categorical variables
le = LabelEncoder()
columns_to_encode = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation", "status"]
for col in columns_to_encode:
    data1[col] = le.fit_transform(data1[col])

print(data1.head())  # Verify the encoded data

# Split the data into features (X) and target (y)
X = data1.iloc[:, :-1]  # All columns except 'status'
y = data1["status"]  # The target column
print("Features (X):\n", X.head())
print("Target (y):\n", y.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train a logistic regression model
lr = LogisticRegression(solver="liblinear")
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)
print("Predictions on test data:\n", y_pred)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the model:", accuracy)

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)

classification_report1 = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report1)

# Make a prediction with a new sample
sample = [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]
sample_df = pd.DataFrame(sample)
print("Prediction for the sample input:", sample_df)


```

## Output:
```
   sl_no gender  ssc_p    ssc_b  ...  specialisation  mba_p      status    salary
0      1      M  67.00   Others  ...          Mkt&HR  58.80      Placed  270000.0
1      2      M  79.33  Central  ...         Mkt&Fin  66.28      Placed  200000.0
2      3      M  65.00  Central  ...         Mkt&Fin  57.80      Placed  250000.0
3      4      M  56.00  Central  ...          Mkt&HR  59.43  Not Placed       NaN
4      5      M  85.80  Central  ...         Mkt&Fin  55.50      Placed  425000.0

[5 rows x 15 columns]
  gender  ssc_p    ssc_b  hsc_p  ... etest_p specialisation  mba_p      status
0      M  67.00   Others  91.00  ...    55.0         Mkt&HR  58.80      Placed
1      M  79.33  Central  78.33  ...    86.5        Mkt&Fin  66.28      Placed
2      M  65.00  Central  68.00  ...    75.0        Mkt&Fin  57.80      Placed
3      M  56.00  Central  52.00  ...    66.0         Mkt&HR  59.43  Not Placed
4      M  85.80  Central  73.60  ...    96.8        Mkt&Fin  55.50      Placed

[5 rows x 13 columns]
Missing values per column:
 gender            0
ssc_p             0
ssc_b             0
hsc_p             0
hsc_b             0
hsc_s             0
degree_p          0
degree_t          0
workex            0
etest_p           0
specialisation    0
mba_p             0
status            0
dtype: int64
Number of duplicate rows: 0
   gender  ssc_p  ssc_b  hsc_p  ...  etest_p  specialisation  mba_p  status
0       1  67.00      1  91.00  ...     55.0               1  58.80       1
1       1  79.33      0  78.33  ...     86.5               0  66.28       1
2       1  65.00      0  68.00  ...     75.0               0  57.80       1
3       1  56.00      0  52.00  ...     66.0               1  59.43       0
4       1  85.80      0  73.60  ...     96.8               0  55.50       1

[5 rows x 13 columns]
Features (X):
    gender  ssc_p  ssc_b  hsc_p  ...  workex  etest_p  specialisation  mba_p
0       1  67.00      1  91.00  ...       0     55.0               1  58.80
1       1  79.33      0  78.33  ...       1     86.5               0  66.28
2       1  65.00      0  68.00  ...       0     75.0               0  57.80
3       1  56.00      0  52.00  ...       0     66.0               1  59.43
4       1  85.80      0  73.60  ...       0     96.8               0  55.50

[5 rows x 12 columns]
Target (y):
 0    1
1    1
2    1
3    0
4    1
Name: status, dtype: int64
Predictions on test data:
 [0 1 1 0 1 0 1 1 1 1 1 1 1 1 0 0 1 0 0 1 0 1 1 1 0 0 1 1 1 1 1 1 1 0 0 1 1
 1 1 1 0 0 1]
Accuracy of the model: 0.813953488372093
Confusion Matrix:
 [[11  5]
 [ 3 24]]
Classification Report:
               precision    recall  f1-score   support

           0       0.79      0.69      0.73        16
           1       0.83      0.89      0.86        27

    accuracy                           0.81        43
   macro avg       0.81      0.79      0.80        43
weighted avg       0.81      0.81      0.81        43

Prediction for the sample input:    0   1   2   3   4   5   6   7   8   9   10  11
0   1  80   1  90   1   1  90   1   0  85   1  85
```

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
