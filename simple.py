import pandas as  pd
from sklearn.tree import DecisionTreeClassifier
house_data = pd.read_csv('house_prices.csv')
#house_data.describe()
X = house_data.drop(columns=["Price"])
y = house_data['Price']
model = DecisionTreeClassifier()
model.fit(X,y)
predictions = model.predict([[4000,5,8]])
result = predictions[0]
result
