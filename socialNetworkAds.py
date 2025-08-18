#import the libs
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pickle import dump

# load the data
data = pd.read_csv("sna_aug25.csv")
# print(data)

# features and target
features = data[["Gender", "Age", "EstimatedSalary"]]
target = data["Purchased"]
# print(features)
# print(target)

# handle cat data
nfeatures = pd.get_dummies(features)
print(nfeatures)

# mms = MinMaxScaler()                    # 82 80 81 77 87 77     ==> 80.667 avg
# sfeatures = mms.fit_transform(nfeatures, target)
# print(sfeatures)

ss = StandardScaler()                     # 85 80 78 84 83 83     ==> 82.166 avg
sfeatures = ss.fit_transform(nfeatures.values)
print(sfeatures)

f = open("scaler.pkl", "wb")
dump(ss, f)
f.close()

# training and testing 
x_train,x_test,y_train,y_test = train_test_split(sfeatures, target)

# model selection
model = LogisticRegression(max_iter=100)
model.fit(x_train, y_train)

# preformance
cm = confusion_matrix(y_test, model.predict(x_test))
cr = classification_report(y_test, model.predict(x_test))
print(cm)
print(cr)

# save the model
f = open("sna.pkl", "wb")
dump(model, f)
f.close()
print("Model saved as sna.pkl")
