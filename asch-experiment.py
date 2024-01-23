import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

csvd = pd.read_csv('asch.csv', skiprows=[0])
dataframe = pd.DataFrame(csvd)
null_counter = dataframe.isnull().sum()
print('Number of Null values in columns:', null_counter)

# dropping null items out of dataframe
prossed_data_with_null = dataframe.convert_dtypes()
prossed_data1 = prossed_data_with_null.dropna()
print(prossed_data1)

# taking out the "gp_error" (which is the Target) from the full dataset
y = prossed_data1[['6']]
# making a new dataset, in which the is no "hmt_score" and str label (which is also called Features data)
X0 = prossed_data1.drop('0', axis=1)
X = X0.drop('6', axis=1)
# test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Decision tree
Tree = DecisionTreeClassifier()
start_train_time = time.time()
Tree.fit(X_train, y_train)
stop_train_time = time.time()
print('Decision Tree Train Time: ', stop_train_time - start_train_time)
start_test_time = time.time()
pred1 = Tree.predict(X_test)
stop_test_time = time.time()
print('Decision Tree Test Time: ', stop_test_time - start_test_time)
print("gp_error (TARGET) Prediction: ", pred1)
msedt = mean_squared_error(y_test, pred1)
print("decision tree mse: ", msedt)

#  kNeighbours
knn = KNeighborsClassifier()
start_train_time2 = time.time()
knn.fit(X_train, y_train)
print('KNN Train Time: ', time.time() - start_train_time2)
start_test_time2 = time.time()
pred2 = knn.predict(X_test)
stop_test_time2 = time.time()
print('KNN Test Time: ', time.time() - start_test_time2)
print("gp_error (TARGET) prediction: ", pred2)
mseknn = mean_squared_error(y_test, pred2)
print("knn mse: ", mseknn)

# Linear Regression
reg = LinearRegression()
reg.fit(X_train, y_train)
pred3 = reg.predict(X_test)
msereg = mean_squared_error(y_test, pred3)
print("regression mse: ", msereg)

# client`s birth year plot
clientBirth = prossed_data1[['7']]
plot0 = plt.boxplot(clientBirth)
plt.show()
# YOU CAN ADD THE PREDICTION RESULTS TO "GP ERROR" LABEL IN ORIGINAL CSV TO SEE HOW THE RESULTS WOULD BE IN DT, KNN, Linear REG SENARIOS
