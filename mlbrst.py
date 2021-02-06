import warnings
warnings.filterwarnings('ignore') #In case any issue occur just ignore
import numpy as np 
import pandas as pd #to read the data
import matplotlib.pyplot as plt #this is for visualization
import seaborn as sns

df = pd.read_csv("https://raw.githubusercontent.com/ingledarshan/AIML-B2/main/data.csv") #this is the link where the data set exists 
df.head()      #it will shows the data (head function show a sample data line)
df.columns     #it will shows the column name of dataset
df.info()       #there is an unnamed column name we need to drop we not need to drop
df['Unnamed: 32'] #this is the column which is ntng we have to drop
df = df.drop("Unnamed: 32", axis=1)#we should drop from the dataset and should be saved in the df
df.head()
df.columns
df.drop('id', axis=1, inplace=True)#we should drop  id colun bcz of no use ,here axis is used of column , inplace is used it should removed from original dataset
#or df = df.drop("id", axis=1)
df.columns
type(df.columns)
l = list(df.columns) #lists all the columns name
 #there are 3 feature so we should separate
features_mean = l[1:11] #mean feature separate

features_se = l[11:21] #se feature separate

features_worst = l[21:] #worst feature separate
print(features_mean) #feature is called as column in machine
print (features_se)
print(features_worst)
df.head(2)
df['diagnosis'].unique() #shows the unique name present in that column
#m =maligant, b = benigns
df['diagnosis'].value_counts() #shows how much count of maligant and benigns
df.describe()# summary of all the numeric columns
#shows the value average (total value), mean deviation (may be between the value of count)
len(df.columns)
#corelation part
corr = df.corr()

plt.figure(figsize=(8,8))
sns.heatmap(corr); #the coorelation data will show in a heatmap

df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})#we should target with number here diagnosis is the target m,B by replacing m=1 b=2
#he diagnosis is dependent all other are independent
X = df.drop('diagnosis', axis=1)
X.head()#here we will separate dependent and independent
Y= df['diagnosis']
Y.head()#here we will separate dependent and independent

#now we will separate the whole x,y data into two part X=xtrain ,xtest and y=ytrain,ytest
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)#here is divided 30% and 70%
#HERE the data has diff range for every colum so we have to compress
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train) #fit(study) and store in mind(x_train) 
X_test = ss.transform(X_test) #and immediatly transform(take test)

#MACHINE LEARNING MODELS
#here we just see implementation and intrepret ....

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression 
lr = LogisticRegression()
lr.fit(X_train, Y_train) #done studing

Y_pred = lr.predict(X_test) #take test by inputting

#now you are going to pridicted and there are many values to correction so we will take out as occuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, Y_pred))

#just for printing
lr_acc = accuracy_score(Y_test, Y_pred)
print(lr_acc)

#WE WILL CREATE A RESULT IN DATA FORM
results = pd.DataFrame()
results

#WE WILL CREATE LIKE A TABLE
tempResults = pd.DataFrame({'Algorithm':['Logistic Regression Method'], 'Accuracy':[lr_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results

# DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)
Y_pred = dtc.predict(X_test)
Y_pred
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, Y_pred))
dtc_acc = accuracy_score(Y_test, Y_pred)
print(dtc_acc)
tempResults = pd.DataFrame({'Algorithm':['Decision tree Classifier Method'], 'Accuracy':[dtc_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results

#RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)
Y_pred = rfc.predict(X_test)
Y_pred
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, Y_pred))
rfc_acc = accuracy_score(Y_test, Y_pred)
print(rfc_acc)
tempResults = pd.DataFrame({'Algorithm':['Random Forest Classifier Method'], 'Accuracy':[rfc_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results

#SUPPORT VECTOR CLASSIFIER
from sklearn import svm
svc = svm.SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
Y_pred
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, Y_pred))
svc_acc = accuracy_score(Y_test, Y_pred)
print(svc_acc)
tempResults = pd.DataFrame({'Algorithm':['Support Vector Classifier Method'], 'Accuracy':[svc_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results