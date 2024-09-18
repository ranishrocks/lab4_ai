# Install pgmpy if not already installed
!pip install pgmpy

# Import necessary libraries
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Step 1: Load the dataset

data = pd.read_csv('/content/InputFile.csv')

# Step 2: Data Preprocessing
# Replace '?' with NaN to handle missing values
data.replace('?', pd.NA, inplace=True)

# Convert relevant columns to numeric, forcing errors to NaN (for columns like TSH, T3, TT4)
data['TSH'] = pd.to_numeric(data['TSH'], errors='coerce')
data['T3'] = pd.to_numeric(data['T3'], errors='coerce')
data['TT4'] = pd.to_numeric(data['TT4'], errors='coerce')

# Fill missing values with the mean for numeric features
data['TSH'].fillna(data['TSH'].mean(), inplace=True)
data['T3'].fillna(data['T3'].mean(), inplace=True)
data['TT4'].fillna(data['TT4'].mean(), inplace=True)

# Convert categorical columns to numeric
data['sex'] = data['sex'].map({'M': 1, 'F': 0})
data['on_thyroxine'] = data['on_thyroxine'].map({'t': 1, 'f': 0})
data['goitre'] = data['goitre'].map({'t': 1, 'f': 0})

# For the target label 'Class', convert 'negative' to 0 and 'positive' to 1
# If you have other label classes, modify this accordingly
data['Class'] = data['Class'].map({'negative': 0, 'positive': 1})

# Step 3: Select relevant features and labels
features = ['age', 'sex', 'on_thyroxine', 'TSH', 'T3', 'TT4', 'goitre']
X = data[features]
y = data['Class']

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build the Bayesian Network
# Define the structure of the Bayesian Network
# Add the 'on_thyroxine' node to the model structure
model = BayesianNetwork([('TSH', 'Class'), ('T3', 'Class'), ('goitre', 'Class'), ('age', 'Class'), ('sex', 'Class'), ('on_thyroxine', 'Class')])

# Combine X_train and y_train for fitting
train_data = X_train.copy()
train_data['Class'] = y_train

# Fit the Bayesian Network using Maximum Likelihood Estimation
model.fit(train_data, estimator=MaximumLikelihoodEstimator)

# Step 6: Make Predictions
# Create inference object
inference = VariableElimination(model)

# Predict for each instance in the test set
predictions = []
for _, row in X_test.iterrows():
    evidence = row.to_dict()
    prediction = inference.map_query(variables=['Class'], evidence=evidence)['Class']
    predictions.append(prediction)

# Convert predictions to pandas series
predictions = pd.Series(predictions)

# Step 7: Evaluate the Model
# Calculate the accuracy score
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")
