import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import time

raw_data = pd.read_csv(r"C:\Users\source\Documents\py\MuhammadRahimi-00130085907004-DataAnalysisProject\asch.csv")
null_count = raw_data.isnull().sum()
print('Number of Null values:', null_count)
# replacing the missing values using default interpolate method (linear); suggest u to read about other methods as well
prossed_data = raw_data.interpolate()
# taking out the "result error label"(which is the Target) from the full dataset
Target = prossed_data[['gp error']]
# making a new dataset, in which the is no "error" label (which is also called Features data)
Features = list(prossed_data.columns)
# test and train
Features_train, Features_test, Target_train, Target_test = train_test_split(Features, Target, shuffle=False)
# Decision tree
Tree = DecisionTreeClassifier()
start_train_time = time.time()
Tree.fit(Features_train, Target_train)
stop_train_time = time.time()
print('Train Time: ', stop_train_time - start_train_time)
predict = Tree.predict(Target_test)
print(predict)






