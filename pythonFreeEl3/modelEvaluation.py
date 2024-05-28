import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = "vgsales_cleaned.csv"
df = pd.read_csv(file_path)

# Assuming 'Global_Sales' is the target variable
# and the rest are feature variables
X = df.drop('Global_Sales', axis=1)
y = df['Global_Sales']

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing for numerical data: impute missing values
numerical_transformer = SimpleImputer(strategy='mean')

# Preprocessing for categorical data: one-hot encode
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
svm_model = SVR(kernel='rbf', C=1.0, epsilon=0.2)

# Create and evaluate the pipeline
pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor), ('model', random_forest_model)])
pipeline_svm = Pipeline(steps=[('preprocessor', preprocessor), ('model', svm_model)])

# Train models
pipeline_rf.fit(X_train, y_train)
pipeline_svm.fit(X_train, y_train)

# Evaluate models
rf_y_pred = pipeline_rf.predict(X_test)
svm_y_pred = pipeline_svm.predict(X_test)

rf_mse = mean_squared_error(y_test, rf_y_pred)
svm_mse = mean_squared_error(y_test, svm_y_pred)

rf_r2 = r2_score(y_test, rf_y_pred)
svm_r2 = r2_score(y_test, svm_y_pred)

print("Random Forest Regressor:")
print(f"MSE: {rf_mse}")
print(f"R^2 Score: {rf_r2}")

print("\nSupport Vector Regressor:")
print(f"MSE: {svm_mse}")
print(f"R^2 Score: {svm_r2}")
