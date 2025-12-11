import pickle
import numpy as np
with open('data/train_data.pkl','rb') as f:
    data = pickle.load(f)
labels = data['labels'].reshape(-1)
print('num samples', labels.shape[0])
print('unique labels', np.unique(labels))
counts = np.bincount(labels)
print('counts', counts)
print('class balance', counts / counts.sum())
