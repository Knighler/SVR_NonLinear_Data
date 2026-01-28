import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model as lm
from sklearn.feature_selection import RFE, f_regression
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor, StackingRegressor, VotingRegressor
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBRegressor

# Reading the data and Classifying 
des = pd.read_csv("D:/asu/fall 24/AI/LinearRegression_HandsOn/Project/train.csv")
test_data = pd.read_csv("D:/asu/fall 24/AI/LinearRegression_HandsOn/Project/test.csv")

des['X3'] = des['X3'].replace('low fat', 'Low Fat')
des['X3'] = des['X3'].replace('LF', 'Low Fat')
des['X3'] = des['X3'].replace('reg', 'Regular')

Q1 = des['X4'].quantile(0.25)
Q3 = des['X4'].quantile(0.75)
lower_bound = Q3 - 1.5 * (Q3 - Q1)
upper_bound = Q3 + 1.5 * (Q3 - Q1)

des['X4'] = des['X4'].apply(lambda x: Q3 if x > upper_bound else x)

# Numerical encoding
normalizer = MinMaxScaler()
encoded_normalized_columns = ['X6','X4']
des[encoded_normalized_columns] = normalizer.fit_transform(des[encoded_normalized_columns])

scalerX2 = StandardScaler()
des['X2'] = scalerX2.fit_transform(des[['X2']])

# String encoding for imputer to work
label_encoder = LabelEncoder()
columns_to_label_encode = ['X1', 'X3', 'X5', 'X7', 'X8', 'X9', 'X10', 'X11']
label_encoders = {}

for col in columns_to_label_encode:
    le = LabelEncoder()
    des[col] = le.fit_transform(des[col])
    label_encoders[col] = le

# Initialize the KNN imputer
imputer = KNNImputer(n_neighbors=8)
des = pd.DataFrame(imputer.fit_transform(des), columns=des.columns)
#des['X1'] = label_encoders['X1'].inverse_transform(des['X1'].astype(int))

# Replacing outliers


# Encoding categorical features with one hot
columns_to_exclude = ['X2', 'X4', 'X6', 'X10', 'Y','X1']  # List of columns to exclude
columns_to_encode = [col for col in des.columns if col not in columns_to_exclude]
columns_to_encode.append('X8')

encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_columns = encoder.fit_transform(des[columns_to_encode])
encoded_columns_df = pd.DataFrame(
    encoded_columns,
    columns=encoder.get_feature_names_out(columns_to_encode),
    index=des.index
)
des = des.drop(columns=columns_to_encode)
des = pd.concat([des, encoded_columns_df], axis=1)
#bINARY ENCODING

def binary_encode(x):
    return [list(format(int(val), 'b').zfill(12)) for val in x]  # You can adjust the number of bits here

binary_encoded_X1 = binary_encode(des['X1'])

binary_encoded_df = pd.DataFrame(binary_encoded_X1, columns=[f'X1_bit_{i+1}' for i in range(len(binary_encoded_X1[0]))], index=des.index)

des = des.drop(columns=['X1'])
des = pd.concat([des, binary_encoded_df], axis=1)

# Check the updated DataFrame








''' **************************************************More Graphs***************************************************
plt.figure(figsize=(10, 6))
sns.boxplot(x=test_data['X4'], color='lightblue') 


sns.stripplot(x=test_data['X4'], color='red', jitter=True)
plt.title('Boxplot of X1 with Points')
plt.xlabel('X1')
plt.grid(True)
plt.show()
#correlation
'''


#***********************************Took the Features with highest correlation(above 0.2 Ig I don't remember)*******************
"""correlation_matrix = des.corr()
threshold = 0.1


strong_correlations = correlation_matrix.where(abs(correlation_matrix) > threshold)

strong_correlations = strong_correlations[~strong_correlations['Y'].isna()]

# Step 3: Drop columns and rows that contain only NaN values
strong_correlations = strong_correlations.dropna(axis=0, how='all')
strong_correlations = strong_correlations.dropna(axis=1, how='all')

print(strong_correlations)
"""
"""
selected_columns = ['X2', 'X4', 'X6', 'X8','X10']


correlations = {}
for col in selected_columns:
    correlations[col] = des[col].corr(des['Y'])

# Print correlations
for col, corr_value in correlations.items():
    print(f"Correlation between {col} and Y: {corr_value}")
print(des['X10'].corr(des['Y']))
"""



#x=des[['X4','X6','X7_OUT019','X11_Supermarket Type1','X11_Supermarket Type3','X9_Small','X8','X10']]

# Define the words to filter *********************************Categorical Data that have weak correlation**********************


word_to_filter1 = 'X3'
word_to_filter2 = 'X5'
#word_to_filter3 = 'X11'
word_to_filter4 = 'X1'
#word_to_filter5 = 'X9'
word_to_filter6 = 'X10'


# Create lists of columns to drop
columns_to_drop1 = [col for col in des.columns if word_to_filter1 in col]
columns_to_drop2 = [col for col in des.columns if word_to_filter2 in col]
#columns_to_drop3 = [col for col in des.columns if word_to_filter3 in col]
columns_to_drop4 = [col for col in des.columns if word_to_filter4 in col]
#columns_to_drop5 = [col for col in des.columns if word_to_filter5 in col]
columns_to_drop6 = [col for col in des.columns if word_to_filter6 in col]

# Combine all columns to drop *********************************Try different Combiantions. This worked the best for XGBoost**************************
columns_to_drop = (
    columns_to_drop1 +
    columns_to_drop2 +
    columns_to_drop4 +
    #columns_to_drop5+
    columns_to_drop6+
    ['X2']+['X4']+
    ['Y']  
)


columns_to_include = ['X6','X4' ,'X7_4.0',"X7_5.0","X8_3.0","X9_1.0","X11_1.0","X11_3.0",'X9_1.0','X7_8.0','X1_bit_10','X8_4.0',
       'X8_5.0', 'X8_6.0','X8_1.0', 'X8_2.0','X1_bit_7', 'X1_bit_11']

x = des.drop('Y', axis=1)


y=des['Y']
f_values, p_values = f_regression(x, y)

# Print F-values and p-values for each feature
"""
for feature, f_val, p_val in zip(x.columns, f_values, p_values):
    print(f"Feature: {feature}, F-value: {f_val:.2f}, p-value: {p_val:.5f}")
"""
# Select features with p-value < 0.05
significant_features = x.columns[p_values < 0.025]
#print("Selected features:", significant_features)
x=x[columns_to_include]




X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001, random_state=42)


""" *************************************Used Linear Regression For Testing*************************************** 
model = lm.LinearRegression()
model.fit(X_train, y_train)"""



# ******************************************************Train XGBoost Model***************************************************
svr_model = SVR(
    kernel='rbf',        # Radial Basis Function kernel
    C=10,              # Regularization parameter
    gamma=0.1,           # Influence of a single training example
    epsilon=0.01,
    degree=3        # Margin of tolerance
)

#xgboost for comparison
xgb_model = XGBRegressor(
    n_estimators=300,  # Number of trees
    learning_rate=0.1,  # Step size shrinkage
    max_depth=3,  # Maximum depth of trees
    random_state=42,
    colsample_bytree= 0.6,
    gamma= 0.5,
    subsample = 1,
    alpha=3


)


param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5, 1],
    'gamma': ['scale', 'auto'],  
    'kernel': ['rbf', 'linear', 'poly'] ,'degree': [2, 3, 4,5]

}


# Train the SVR model
svr_model.fit(X_train, y_train)
svr_predictions_train = svr_model.predict(X_train)
"""
svr_model = SVR(kernel='rbf', C=10, gamma=0.1, epsilon=0.01)
cv_scores = cross_val_score(svr_model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')

print("Cross-validated MAE:", -cv_scores.mean())"""

# Calculate residuals (y - predictions) for the training set
residuals_train = y_train - svr_predictions_train

# Train Ridge Regression on residuals
ridge_model = Ridge(alpha=5)  # Regularization parameter; tune this if needed
ridge_model2 = Ridge(alpha=2) 

ridge_model.fit(X_train, residuals_train)


# Ridge model's prediction for residuals on test data
ridge_predictions_test = ridge_model.predict(X_test)

all_predictions = ridge_predictions_test + svr_model.predict(X_test)
mae=mean_absolute_error(y_test, all_predictions)

print("ridge mae:",mae)
print("svr: ",mean_absolute_error(y_test,svr_model.predict(X_test)))




X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_train = np.array(X_train)
xgb_model.fit(X_train, y_train)


X_test = X_test.apply(pd.to_numeric, errors='coerce')
X_test = np.array(X_test)
xgb_model_predictions_train = xgb_model.predict(X_test)

residuals_train2 = y_train - xgb_model.predict(X_train)
ridge_model2.fit(X_train, residuals_train2)
all_predictions2 = ridge_predictions_test + xgb_model.predict(X_test)

print("xgb: ",mean_absolute_error(y_test,xgb_model_predictions_train))
print("xgb with ridge: ",mean_absolute_error(y_test, all_predictions2))
##########################################################################################################Test Data###################################################3

test_data['X3'] = test_data['X3'].replace('low fat', 'Low Fat')
test_data['X3'] = test_data['X3'].replace('LF', 'Low Fat')
test_data['X3'] = test_data['X3'].replace('reg', 'Regular')

#numerical encoding
normalizer=MinMaxScaler()
encoded_normalized_columns=['X4','X6']
test_data[encoded_normalized_columns]=normalizer.fit_transform(test_data[encoded_normalized_columns])

scalerX2=StandardScaler()
test_data['X2']=scalerX2.fit_transform(test_data[['X2']])
#String encoding for imputer to work

label_encoder = LabelEncoder()
columns_to_label_encode = ['X1', 'X3', 'X5', 'X7', 'X8', 'X9', 'X10', 'X11']
label_encoders = {}

for col in columns_to_label_encode:
    le = LabelEncoder()
    test_data[col] = le.fit_transform(test_data[col])
    label_encoders[col] = le
    



# Initialize the KNN imputer
imputer = KNNImputer(n_neighbors=5)  # Adjust `n_neighbors` as needed
test_data = pd.DataFrame(imputer.fit_transform(test_data), columns=test_data.columns)



#replacing outliers
Q1 = test_data['X4'].quantile(0.25)
Q3 = test_data['X4'].quantile(0.75)
lower_bound = Q3 - 1.5 * (Q3 - Q1)
upper_bound = Q3 + 1.5 * (Q3 - Q1)


test_data['X4'] = test_data['X4'].apply(lambda x:Q3 if x > upper_bound else x)

columns_to_exclude = ['X2', 'X4', 'X6','X10','Y','X1']  # List of columns to exclude
columns_to_encode = [col for col in test_data.columns if col not in columns_to_exclude]
columns_to_encode.append('X8')

encoder = OneHotEncoder(sparse_output=False,drop='first')
encoded_columns = encoder.fit_transform(test_data[columns_to_encode])
encoded_columns_df=pd.DataFrame(encoded_columns,columns=encoder.get_feature_names_out(columns_to_encode))
encoded_columns_df = pd.DataFrame(
    encoded_columns,
    columns=encoder.get_feature_names_out(columns_to_encode),
    index=test_data.index
)
test_data = test_data.drop(columns=columns_to_encode)

test_data = pd.concat([test_data, encoded_columns_df], axis=1)

def binary_encode(x):
    return [list(format(int(val), 'b').zfill(12)) for val in x]  # You can adjust the number of bits here

binary_encoded_X1 = binary_encode(test_data['X1'])

binary_encoded_df = pd.DataFrame(binary_encoded_X1, columns=[f'X1_bit_{i+1}' for i in range(len(binary_encoded_X1[0]))], index=test_data.index)

test_data = test_data.drop(columns=['X1'])
test_data = pd.concat([test_data, binary_encoded_df], axis=1)

real_x = test_data[x.columns]
real_x = real_x.loc[:, ~real_x.columns.duplicated()]
real_x = real_x[x.columns]
svr_predictions_test = svr_model.predict(real_x)
ridge_predictions_test = ridge_model.predict(real_x)
final_predictions = svr_predictions_test 


#xgboost

real_x = test_data[x.columns]
real_x = real_x.loc[:, ~real_x.columns.duplicated()]
real_x = real_x[x.columns]



sample = pd.read_csv('D:/asu/fall 24/AI/LinearRegression_HandsOn/Project/sample_submission.csv')
sample['Y'] = final_predictions
sample.to_csv('D:/asu/fall 24/AI/LinearRegression_HandsOn/Project/SVR_KNNImputer_MoreFeatures_ANOVA2.csv', index=False)
compare_sample = pd.read_csv('D:/asu/fall 24/AI/LinearRegression_HandsOn/Project/SVR_KNNImputer_MoreFeatures_ANOVA.csv')
print("done")
print(mean_absolute_error(final_predictions,compare_sample['Y'])) 


"""
real_x = real_x.apply(pd.to_numeric, errors='coerce')
real_x = np.array(real_x)

xgb_predictions_test = xgb_model.predict(real_x)
sample = pd.read_csv('D:/asu/fall 24/AI/LinearRegression_HandsOn/Project/sample_submission.csv')
sample['Y'] = xgb_predictions_test
sample.to_csv('D:/asu/fall 24/AI/LinearRegression_HandsOn/Project/SVR_KNNImputer_MoreFeatures_Ridge.csv', index=False)
compare_sample = pd.read_csv('D:/asu/fall 24/AI/LinearRegression_HandsOn/Project/SVR_KNNImputer_MoreFeatures.csv')
print("done")
print(mean_absolute_error(xgb_predictions_test,compare_sample['Y']))"""