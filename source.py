
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
print(1)
# Load the data into Pandas dataframes
train_df = pd.read_csv(r"C:\Users\91892\Desktop\Coding Block\Dataset\train.csv")
test_df = pd.read_csv(r"C:\Users\91892\Desktop\Coding Block\Dataset\test.csv")

# Preprocess the data
print(2)
# Combine text columns into a single column
train_df['text'] = train_df['TITLE'] + ' ' + train_df['DESCRIPTION'] + ' ' + train_df['BULLET_POINTS']
test_df['text'] = test_df['TITLE'] + ' ' + test_df['DESCRIPTION'] + ' ' + test_df['BULLET_POINTS']
print(3)
# Remove punctuation,special characters, and stop words from the text column
vectorizer = TfidfVectorizer(decode_error='replace', encoding='utf-8')
train_text = vectorizer.fit_transform(train_df['text'].apply(lambda train_text: np.str_(train_text)))
test_text = vectorizer.fit_transform(test_df['text'].apply(lambda test_text: np.str_(test_text)))
#test_text = vectorizer.transform(test_df['text'])
print(4)
# Convert the categorical variable (product type ID) into numerical variables using one-hot encoding
one_hot_encoder = OneHotEncoder()
train_product_type = one_hot_encoder.fit_transform(train_df['PRODUCT_TYPE_ID'].values.reshape(-1, 1))
test_product_type = one_hot_encoder.transform(test_df['PRODUCT_TYPE_ID'].values.reshape(-1, 1))
print(5)
# Combine the processed text and numerical variables into a single feature matrix
train_features = np.hstack((train_text.toarray(), train_product_type.toarray()))
test_features = np.hstack((test_text.toarray(), test_product_type.toarray()))
print(6)
# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_features, train_df['PRODUCT_LENGTH'], test_size=0.2, random_state=42)
print(7)
# Train and evaluate a random forest regression model
rf_reg = RandomForestRegressor()
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_val)
print('Validation MSE:', mean_squared_error(y_val, y_pred))
print('Validation MAE:', mean_absolute_error(y_val, y_pred))
print(8)
# Fine-tune the model using cross-validation
param_grid = {'n_estimators': [100, 200, 500], 'max_depth': [10, 20, 30]}
grid_search = GridSearchCV(rf_reg, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print('Best parameters:', grid_search.best_params_)
y_pred = grid_search.predict(X_val)
print('Validation MSE:', mean_squared_error(y_val, y_pred))
print('Validation MAE:', mean_absolute_error(y_val, y_pred))
print(9)
# Train the final model on the entire training set and make predictions on the testing set
rf_reg = RandomForestRegressor(n_estimators=grid_search.best_params_['n_estimators'], max_depth=grid_search.best_params_['max_depth'])
rf_reg.fit(train_features, train_df['PRODUCT_LENGTH'])
test_pred = rf_reg.predict(test_features)
print(10)
# Save the predictions in a CSV file
submission_df = pd.DataFrame({'PRODUCT_ID': test_df['PRODUCT_ID'], 'PRODUCT_LENGTH': test_pred})
submission_df.to_csv('C:/Users/91892/Desktop/Coding Block/Dataset/submission.csv',index=False)
print(11)