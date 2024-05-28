import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = "vgsales_cleaned.csv"
df = pd.read_csv(file_path)

# Adjust display settings to prevent truncation
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Display the first few rows of the dataframe
print("DataFrame Head:\n", df.head())

# Assuming 'Global_Sales' is the target variable
# and the rest are feature variables
X = df.drop('Global_Sales', axis=1)
y = df['Global_Sales']

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing for numerical data: impute missing values and scale
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data: one-hot encode and impute missing values
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define the model
model = SVR(kernel='rbf', C=1.0, epsilon=0.2)

# Create and evaluate the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)
                          ])

# Perform cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
cv_mse = -cv_scores.mean()

print(f"Cross-Validation Mean Squared Error: {cv_mse}")

# Fit the model to the entire dataset
pipeline.fit(X, y)

# Make predictions on the entire dataset
y_pred = pipeline.predict(X)

# Evaluate the model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Mean Squared Error on Entire Dataset: {mse}")
print(f"R^2 Score on Entire Dataset: {r2}")
