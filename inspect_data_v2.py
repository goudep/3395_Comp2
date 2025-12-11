
import pickle
import numpy as np

def inspect_data():
    try:
        with open('data/train_data.pkl', 'rb') as f:
            train_data = pickle.load(f)
        with open('data/test_data.pkl', 'rb') as f:
            test_data = pickle.load(f)
            
        print("Train Data Keys:", train_data.keys())
        print("Test Data Keys:", test_data.keys())
        
        train_images = train_data['images']
        train_labels = train_data['labels']
        test_images = test_data['images']
        
        print("Train Images Shape:", train_images.shape)
        print("Train Labels Shape:", train_labels.shape)
        print("Test Images Shape:", test_images.shape)
        
        print("Train Images Dtype:", train_images.dtype)
        print("Train Images Min/Max:", train_images.min(), train_images.max())
        
        unique_labels = np.unique(train_labels)
        print("Unique Labels:", unique_labels)
        print("Class Counts:", np.bincount(train_labels.flatten()))
        
    except Exception as e:
        print(f"Error inspecting data: {e}")

if __name__ == "__main__":
    inspect_data()
