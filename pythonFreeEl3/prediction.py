import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'vgsales_cleaned.csv'
data = pd.read_csv(file_path)

# Encoding categorical features
label_encoders = {}
categorical_columns = ['Name', 'Platform', 'Genre', 'Publisher']

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Features and target variable
features = ['Rank', 'Name', 'Platform', 'Year', 'Genre', 'Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
target = 'Global_Sales'

X = data[features]
y = data[target]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Actual vs Predicted values comparison
comparison_df = X_test.copy()
comparison_df['Actual_Global_Sales'] = y_test.values
comparison_df['Predicted_Global_Sales'] = y_pred

# Decode categorical features for better readability
for col in categorical_columns:
    comparison_df[col] = label_encoders[col].inverse_transform(comparison_df[col])

# Print the comparison without truncation
print("Actual vs Predicted Global Sales:")
print(comparison_df[['Rank', 'Name', 'Platform', 'Year', 'Genre', 'Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Actual_Global_Sales', 'Predicted_Global_Sales']].to_string(index=False))

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(comparison_df['Actual_Global_Sales'], comparison_df['Predicted_Global_Sales'], color='blue')
plt.plot([min(comparison_df['Actual_Global_Sales']), max(comparison_df['Actual_Global_Sales'])], [min(comparison_df['Actual_Global_Sales']), max(comparison_df['Actual_Global_Sales'])], color='red', lw=2)
plt.xlabel('Actual Global Sales')
plt.ylabel('Predicted Global Sales')
plt.title('Actual vs Predicted Global Sales')
plt.grid(True)
plt.show()

# Function to predict the next global sales
def predict_global_sales(new_data):
    # Encode categorical features of the new data
    for col in categorical_columns:
        new_data[col] = label_encoders[col].transform([new_data[col]])[0]

    # Convert the new data to a DataFrame
    new_data_df = pd.DataFrame([new_data])

    # Predict the global sales
    prediction = model.predict(new_data_df)

    return prediction[0]

# Example new data (replace with actual new data)
new_data = {
    'Rank': 9,
    'Name': 'New Game',
    'Platform': 'Wii',
    'Year': 2024,
    'Genre': 'Sports',
    'Publisher': 'Nintendo',
    'NA_Sales': 10.0,
    'EU_Sales': 7.5,
    'JP_Sales': 5.0,
    'Other_Sales': 2.0
}

# Predict the global sales for the new data
predicted_sales = predict_global_sales(new_data)
print(f"\nPredicted Global Sales for the new game: {predicted_sales:.2f} million")
