# --------------
# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report,confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Load the data
#Loading the Spam data from the path variable for the mini challenge
#Target variable is the 57 column i.e spam, non-spam classes 
df = pd.read_csv(path, header=None)

# Overview of the data
#print(df.info())
#print(df.describe())
#Dividing the dataset set in train and test set and apply base logistic model
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
# Calculate accuracy , print out the Classification report and Confusion Matrix.
Accuracy = accuracy_score(y_test, y_pred)
#print(Accuracy)
cr = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

lasso = Lasso()
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
zero_features = np.sum(lasso.coef_==0)
#Accuracy_lasso = accuracy_score(y_test, lasso_pred)
#print(Accuracy_lasso)
#print(zero_features)
# Copy df in new variable df1
df1 = df.copy()
correlation_matrix = df1.drop(57,1).corr()
# Remove Correlated features above 0.75 and then apply logistic model
upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))
columns_to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]
df1.drop(columns_to_drop, axis=1, inplace=True)
# Split the new subset of data and fit the logistic model on training data
X_new = df1.iloc[:,:-1]
y_new = df1.iloc[:,-1]
X_tr1, X_te1, y_tr1, y_te1 = train_test_split(X_new, y_new, test_size=0.3, random_state=1)
lr.fit(X_tr1, y_tr1)
y1_pred = lr.predict(X_te1)

# Calculate accuracy , print out the Classification report and Confusion Matrix for new data
Accuracy_1 = accuracy_score(y_te1, y1_pred)
#print(Accuracy_1)
cr_1 = classification_report(y_te1, y1_pred)
cm_1 = confusion_matrix(y_te1, y1_pred)
# Apply Chi Square and fit the logistic model on train data use df dataset
n_feature = [10,15,20,25,30,35,40,45,50,55]
Highest_Accuracy = 0

for i in n_feature :
 test = SelectKBest(score_func=chi2, k=i)
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

 X_train_transformed = test.fit_transform(X_train, y_train)
 X_test_transformed = test.transform(X_test)

 lr_model_3 = LogisticRegression()
 lr_model_3.fit(X_train_transformed,y_train)
 #y_pred = lr_model_3.predict(X_test_transformed)
 Accuracy_chi = lr_model_3.score(X_test_transformed, y_test)
 if Accuracy_chi >= Highest_Accuracy :
    Highest_Accuracy = Accuracy_chi
    optimal_n = i
 
# Calculate accuracy , print out the Confusion Matrix 
test = SelectKBest(score_func=chi2, k=55)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

X_train_transformed = test.fit_transform(X_train, y_train)
X_test_transformed = test.transform(X_test)

lr_model_3 = LogisticRegression()
lr_model_3.fit(X_train_transformed,y_train)
y_pred = lr_model_3.predict(X_test_transformed)
Accuracy_chi = lr_model_3.score(X_test_transformed, y_test)
cr_chi2 = classification_report(y_test, y_pred)
cm_chi2 = confusion_matrix(y_test, y_pred)
# Apply Anova and fit the logistic model on train data use df dataset
test = SelectKBest(score_func=f_classif, k=55)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

X_train_transformed = test.fit_transform(X_train, y_train)
X_test_transformed = test.transform(X_test)

lr_model_4 = LogisticRegression()
lr_model_4.fit(X_train_transformed,y_train)
y_pred = lr_model_4.predict(X_test_transformed)

# Calculate accuracy , print out the Confusion Matrix 
Accuracy_fscore = lr_model_4.score(X_test_transformed, y_test)
cr_fs = classification_report(y_test, y_pred)
cm_fs = confusion_matrix(y_test, y_pred)

# Apply PCA and fit the logistic model on train data use df dataset
n_components = [10,15,20,25,30,35,40,45,50,55]
Highest_Accuracy = 0

for n in n_components:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    pca = PCA(n_components=n)

    X_train_transformed = pca.fit_transform(X_train)
    X_test_transformed = pca.transform(X_test)

    lr_model_3 = LogisticRegression()
    lr_model_3.fit(X_train_transformed, y_train)
    y_pred = lr_model_3.predict(X_test_transformed)
    PCA_Accuracy = lr_model_3.score(X_test_transformed, y_test)
    if PCA_Accuracy >= Highest_Accuracy :
        Highest_Accuracy = PCA_Accuracy
        Optimal_PCA_Component = n
print("Best PCA Accuracy is ", Highest_Accuracy, "With component ", n)

# Calculate accuracy , print out the Confusion Matrix 
cm_psa = confusion_matrix(y_test, y_pred)
print(cm_psa)
# Compare observed value and Predicted value
print(lr_model_3.predict(X_test_transformed[0:10]))
print(y_test[0:10].values)


