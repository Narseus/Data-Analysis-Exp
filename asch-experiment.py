import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import time
from sklearn.neighbors import KNeighborsClassifier

# opening new csv using pandas and i had to skip header because it wasn't an integer
url = "https://drive.google.com/file/d/10SbEME3_x_Avwcq5azeH6B9hmMaySweu/view?usp=drivesdk"
raw_data = pd.read_csv(url, skiprows =[0])
null_count = raw_data.isnull().sum()
print('Number of Null values in columns:', null_count)

# replacing the missing values using default interpolate method (linear); suggest u to read about other methods as well
prossed_data = raw_data.interpolate()
# taking out the "result error label"(which is the Target) from the full dataset
Y = prossed_data[['7']]
# making a new dataset, in which the is no "error" label (which is also called Features data)
X = prossed_data[['0', '1', '2', '3', '4', '5']]

# test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# Decision tree
Tree = DecisionTreeClassifier()
start_train_time = time.time()
Tree.fit(X_train, Y_train)
stop_train_time = time.time()
print('Decision Tree Train Time: ', stop_train_time - start_train_time)
start_test_time = time.time()
y_pred = Tree.predict(X_test)
stop_test_time = time.time()
print('Decision Tree Test Time: ',stop_test_time - start_test_time)
print("GP-ERROR (TARGET) Prediction: ", y_pred)

#  kNeighbours
knn = KNeighborsClassifier()
start_train_time2 = time.time()
knn.fit(X_train, Y_train)
print('KNN Train Time: ', time.time() - start_train_time2)
start_test_time2 = time.time()
y_pred2 = knn.predict(X_test)
stop_test_time2 = time.time()
print('KNN Test Time: ', time.time() - start_test_time2)
print("GP-ERROR (TARGET) prediction: ", y_pred2)

# YOU CAN ADD THE PREDICTION RESULTS TO "GP ERROR" LABEL IN ORIGINAL CSV TO SEE HOW THE RESULTS WOULD BE IN DT & KNN SENARIOS
