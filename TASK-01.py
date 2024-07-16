import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


print("Training Dataset:")
print(train_df.head())

print("\nTesting Dataset:")
print(test_df.head())


features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']
target = 'SalePrice'

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features] 


print("\nMissing values in training set:")
print(X_train.isnull().sum())
print("\nMissing values in test set:")
print(X_test.isnull().sum())

model = LinearRegression()


model.fit(X_train, y_train)


predictions = model.predict(X_test)

train_predictions = model.predict(X_train)
train_mse = mean_squared_error(y_train, train_predictions)
train_r2 = r2_score(y_train, train_predictions)

print("\nModel Evaluation on Training Set:")
print(f"Train MSE: {train_mse}")
print(f"Train R^2: {train_r2}")


submission_df = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': predictions})
print("\nSample of Predicted Sale Prices:")
print(submission_df.head())


save_path = 'predictions.csv' 
submission_df.to_csv(save_path, index=False)
print(f"\nPredictions saved to {save_path}")
