# Jinhwan Kim - 5838884

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score

# Load the stroke dataset from CSV file
data = pd.read_csv("stroke.csv")

# Encode categorical variables into numbers
label_encoder = LabelEncoder()
# 0 = Male, 1 = Female
data['gender'] = label_encoder.fit_transform(data['gender'])
# 0 = Never Smoked, 1 = Formerly Smoked, 2 = Smokes
data['smoking_status'] = label_encoder.fit_transform(data['smoking_status'])

# Define the features (inputs) and target (output)
X = data[['gender', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'smoking_status']]
y = data['stroke']

# Fill any missing values in the dataset with the average of the respective column
X = SimpleImputer(strategy='mean').fit_transform(X)

# Balance the sample data by up-sampling the high-risk stroke cases to avoid bias
X_high_risk = X[y == 1]
y_high_risk = y[y == 1]
X_not_high_risk = X[y == 0]
y_not_high_risk = y[y == 0]

# Resample the high-risk cases as same occurrence as the not-high-risk cases
X_high_risk_upsampled, y_high_risk_upsampled = resample(
    X_high_risk, y_high_risk,
    replace=True, n_samples=len(y_not_high_risk), random_state=42
)

# Combine the resampled high-risk cases with the low-risk cases
X = np.vstack((X_not_high_risk, X_high_risk_upsampled))
y = np.hstack((y_not_high_risk, y_high_risk_upsampled))

# To help the model understand the data better, scale the features in a similar range
scaler = StandardScaler()
X = scaler.fit_transform(X)

# To help the model consider combined interaction of features, include the polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X = poly.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model to predict high risk vs low risk individuals
# class_weight='balanced' gives equal importance to both stroke and no-stroke cases
model = LogisticRegression(max_iter=500, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate the model by checking accuracy on test data
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Perform cross-validation between multiple trainings/data sets to improve accuracy
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Cross-Validation Accuracy: {cv_scores.mean():.2f}")

# Function to get user inputs to predict stroke risk
def get_user_input():
    # Get valid user input (gender, age, hypertension, etc.) and ensure it's within a valid range.
    gender = get_valid_input("Gender (0 = Male, 1 = Female): ", int, (0, 1))
    age = get_valid_input("Age (0-120): ", float, (0, 120))
    hypertension = get_valid_input("Hypertension (0 = No, 1 = Yes): ", int, (0, 1))
    heart_disease = get_valid_input("Heart Disease (0 = No, 1 = Yes): ", int, (0, 1))
    avg_glucose_level = get_valid_input("Average Glucose Level: (50-300): ", float, (50, 300))
    bmi = get_valid_input("BMI (10-100): ", float, (10, 100))
    smoking_status = get_valid_input("Smoking Status (0 = never smoked, 1 = formerly smoked, 2 = smokes): ", int, (0, 2))

    # Store the user inputs in a list and scale them
    user_data = [[gender, age, hypertension, heart_disease, avg_glucose_level, bmi, smoking_status]]
    user_data = scaler.transform(user_data)
    user_data = poly.transform(user_data)

    return user_data


# Function to predict stroke risk based on user input
def predict_stroke_risk(model):
    user_data = get_user_input()
    # Predict whether the individual is high risk or not
    prediction = model.predict(user_data)
    # Get the probability of stroke
    probability = model.predict_proba(user_data)

    # Output the prediction in a more readable way
    if prediction[0] == 1:
        print("\nHigh Risk")
    else:
        print("\nNot High-Risk")
    print(f"Estimated probability of stroke: {probability[0][1] * 100:.2f}%")


# Feature importance plot indicating the extent of contribution of each feature
def feature_importance_plot():
    feature_names = ['gender', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'smoking_status']

    # The model's coefficients (importance) of each feature
    coef = model.coef_[0][:len(feature_names)]

    # Create a bar chart showing each feature's importance
    # Red = positive correlation, Green = negative correlation
    plt.barh(feature_names, coef, color=['green' if c < 0 else 'red' for c in coef])
    plt.title('Feature Importance: Stroke Risk Prediction')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    # Autofit the layout
    plt.tight_layout()
    plt.show()


# Box plot showing the distribution of glucose levels vs high risk and not high-risk.
def create_glucose_boxplot():
    plt.figure(figsize=(8, 6))
    data.boxplot(column='avg_glucose_level', by='stroke', grid=False, patch_artist=True, showfliers=False)
    plt.title('Boxplot of Average Glucose Level by Stroke Risk')
    plt.suptitle('')
    plt.xlabel('Stroke Risk (0 = Not High-Risk, 1 = High Risk)')
    plt.ylabel('Average Glucose Level')
    # Autofit the layout
    plt.tight_layout()
    plt.show()


# Function to create all visualizations (pie chart, feature importance, and box plot).
def create_visualizations():
    # Pie chart showing the proportion of high risk vs not high-risk
    stroke_counts = data['stroke'].value_counts()
    plt.pie(stroke_counts, labels=['Not High-Risk', 'High Risk'], autopct='%1.1f%%', colors=['green', 'red'])
    plt.title('Proportion of Stroke Risk')
    plt.tight_layout()
    plt.show()
    # Show the other two visualizations
    feature_importance_plot()
    create_glucose_boxplot()


# A function to validate user input is within the range
def get_valid_input(prompt, input_type, valid_range):
    while True:
        try:
            value = input_type(input(prompt))
            if value >= valid_range[0] and value <= valid_range[1]:
                return value
            else:
                print(f"Input must be between {valid_range[0]} and {valid_range[1]}.")
        except ValueError:
            print(f"Invalid input. Please enter a {input_type.__name__}.")


# Main program interface
if __name__ == "__main__":
    while True:
        print("\n1. Predict Stroke Risk")
        print("2. Show Visualizations")
        print("3. Exit")
        choice = input("Enter choice (1/2/3): ")

        if choice == '1':
            predict_stroke_risk(model)
        elif choice == '2':
            create_visualizations()
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice, please try again.")
