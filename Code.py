# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Importing the dataset
df = pd.read_csv('cancer.csv')
df.replace('?', -99999, inplace=True)
df.drop(columns=['id'], inplace=True)

X = np.array(df.drop(['classes'], 1))
y = np.array(df['classes'])

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Principle Component Analysis
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Fitting KNN to the Training set
knn_accuracy = []
for i in range(1, 21):
    classifier = KNeighborsClassifier(n_neighbors=i)
    trained_model = classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    knn_accuracy.append(accuracy_score(y_test, y_pred) * 100)

# Fitting SVM to the Training set
svm_classifier = SVC(kernel='linear', random_state=0)
svm_trained_model = svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm) * 100

# Fitting Decision Tree to the Training set
dt_accuracy = []
for depth in range(1, 21):
    classifier = DecisionTreeClassifier(max_depth=depth, random_state=42)
    trained_model = classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    dt_accuracy.append(accuracy_score(y_test, y_pred) * 100)

# Fitting Random Forest Classifier to the Training set
rf_accuracy = []
for estimators in range(1, 21):
    classifier = RandomForestClassifier(n_estimators=estimators, random_state=42)
    trained_model = classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    rf_accuracy.append(accuracy_score(y_test, y_pred) * 100)

# Fitting Logistic Regression to the Training set
logreg_classifier = LogisticRegression(random_state=0)
logreg_trained_model = logreg_classifier.fit(X_train, y_train)
y_pred_logreg = logreg_classifier.predict(X_test)
logreg_accuracy = accuracy_score(y_test, y_pred_logreg) * 100

# Plotting accuracies for different models
plt.figure(figsize=(12, 6))
plt.plot(range(1, 21), knn_accuracy, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10, label='KNN')
plt.plot(range(1, 21), dt_accuracy, color='green', linestyle='dashed', marker='o',  
         markerfacecolor='orange', markersize=10, label='Decision Tree')
plt.plot(range(1, 21), rf_accuracy, color='purple', linestyle='dashed', marker='o',  
         markerfacecolor='pink', markersize=10, label='Random Forest')
plt.axhline(y=svm_accuracy, color='black', linestyle='dashed', label='SVM')
plt.axhline(y=logreg_accuracy, color='cyan', linestyle='dashed', label='Logistic Regression')
plt.title('Accuracy for different Models and Parameters')
plt.xlabel('Parameter Value')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Displaying accuracy separately
print("Accuracy score of KNN:", knn_accuracy)
print("Accuracy score of Decision Trees:", dt_accuracy)
print("Accuracy score of Random Forest:", rf_accuracy)
print("Accuracy score of test SVM:", svm_accuracy)
print("Accuracy score of test Logistic Regression:", logreg_accuracy)
