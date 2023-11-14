import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from sklearn.metrics import accuracy_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def load_data(paths):
    data = []
    for p in paths:
        with open(p, 'r') as f:
            data += json.load(f)
    return [{'bot': d['bot'], 'user': d['user']} for d in data]

folder_path = 'MMDS/1'  
file_list = []

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename).replace('\\', '/')
    file_list.append(file_path)
df = load_data(file_list)
df = pd.DataFrame(df)
y = df['bot'].astype('int')
X = pd.get_dummies(df['user'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
