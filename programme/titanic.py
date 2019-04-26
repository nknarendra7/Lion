import pandas as pd 

train = 'train.csv'
test = 'test.csv'

train = pd.read_csv(train)

test = pd.read_csv(test)
test = test.fillna(0)
# print(test.info())
combine = [train,test]
for data in combine:
	data['Age'] = data['Age'].fillna(data['Age'].mean())
	data['familly'] = data['SibSp'] + data['Parch']
	data.drop(['SibSp','Parch'],axis = 1,inplace = True)
	data.drop('Cabin',axis = 1,inplace = True)
	data.fillna(data['Embarked'].value_counts().index[0])
	# print(train.info())
	data['Sex'] = data['Sex'].astype('category').cat.codes
	# print(train.info())
	data['Embarked'] = data['Embarked'].astype('category').cat.codes
	# print(train.info())
	data.drop('Ticket',axis = 1,inplace = True)
	data.drop('Name',axis = 1,inplace = True)
for data in combine:
	data.loc[ data['Age'] <= 16,'Age'] = 0
	# print(train['Age'].max())
	data.loc[(data['Age']>16) & (data['Age'] <=32),'Age'] = 1
	data.loc[ (data['Age']>32) & (data['Age']<=64),'Age'] = 2
	data.loc[ data['Age']>64,'Age'] = 3
	data.loc[ data['Fare'] <= 7.91, 'Fare'] = 0
	data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
	data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']   = 2
	data.loc[ data['Fare'] > 31, 'Fare'] = 3
	data['Fare'] = data['Fare'].astype(int)
X_train = train.drop(['Survived',"PassengerId"],axis=1)
Y_train = train['Survived']
X_test = test.drop("PassengerId",axis = 1).copy()
print(X_train.shape,Y_train.shape)
print(X_test.shape)
print(X_train.columns)
print(X_test.columns)
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,Y_train)
Y_pred = log.predict(X_test)
print(round(log.score(X_train,Y_train)*100),2)
