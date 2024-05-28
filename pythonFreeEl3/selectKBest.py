import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
df = pd.read_csv('vgsales_cleaned.csv')

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns

# Apply one-hot encoding to categorical columns
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_categorical = encoder.fit_transform(df[categorical_columns])

# Create a DataFrame for the encoded categorical features
encoded_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_columns))

# Drop the original categorical columns from the dataset
df = df.drop(columns=categorical_columns)

# Concatenate the encoded categorical features with the original DataFrame
df = pd.concat([df, encoded_df], axis=1)

# Define features (X) and target variable (y)
X = df.drop(columns=['Global_Sales'])
y = df['Global_Sales']

# Select the top k features based on their F-statistic scores
select_kbest = SelectKBest(score_func=f_regression, k='all')
fit = select_kbest.fit(X, y)

# Get the scores and corresponding feature names
feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': fit.scores_})

# Sort the features based on their scores
selected_features = feature_scores.sort_values(by='Score', ascending=False)

# Display the selected features
print("Top features selected by SelectKBest:")
print(selected_features)
