import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import warnings
import torch
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

from transformers import AutoTokenizer, AutoModel
import torch
# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Tokenize sentences
encoded_input = tokenizer(df['user'].values.tolist(), padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling. In this case, max pooling.
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
X = pd.DataFrame(sentence_embeddings.numpy())


y = df['bot'].astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LogisticRegression
lrc = LogisticRegression(solver='sag').fit(X_train, y_train)
y_pred = lrc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'The accuracy for model is {accuracy}')