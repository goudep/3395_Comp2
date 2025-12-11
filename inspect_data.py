import pickle
for split in ['train','test']:
    with open(f'data/{split}_data.pkl','rb') as f:
        data = pickle.load(f)
    print(split, type(data))
    if isinstance(data, dict):
        for k,v in data.items():
            print(' ', k, type(v), getattr(v,'shape',None))
    else:
        print(getattr(data,'shape',None))
