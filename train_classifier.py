import pickle 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np 

data_dict = pickle.load(open("data.pickle", 'rb'))

print([len(item) if isinstance(item, list) else 'Not a list' for item in data_dict['data']])  # Check lengths

indices_84 = [i for i, item in enumerate(data_dict['data']) if isinstance(item, list) and len(item) == 84]
for i, item in enumerate(data_dict['data']):
    if isinstance(item, list) and len(item) == 84:
        data_dict['data'][i] = item[:42]  # Truncate to length 42

print("Modified data where length was 84.")


print("Indices of elements with length 84:", indices_84)



data = np.array(data_dict['data'], dtype="object")
labels = np.array(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()