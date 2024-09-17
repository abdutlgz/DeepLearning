import torch
import torch.utils.data
from sklearn.datasets import load_digits
from torch.utils.data import TensorDataset, random_split

def parabola_1d(n_samples):
    """
    1d parabola on uniform grid points.
    """
    x = torch.linspace(-1, 1, 100).reshape((-1,1))
    y = x**2

    dataset = torch.utils.data.TensorDataset(x, y)
    dataset.data = x
    dataset.targets = y

    return dataset

def parabola_nd(n_samples, dim):
    """
    Parabola x[0]^2 + x[1]^2 + ... in dim dimensions.
    """
    # Uniformly sample from [-1, 1]^dim
    x = torch.FloatTensor(n_samples, dim).uniform_(-1, 1)
    
    # Sum of squares along the dimension axis
    y = torch.sum(x**2, dim=1, keepdim=True)
    
    dataset = torch.utils.data.TensorDataset(x, y)
    dataset.data = x
    dataset.targets = y
    
    return dataset

def load_digits_data(test_split=0.2):
    digits = load_digits()
    #using hint
    X = digits.data.astype('float32')  
    y = digits.target.astype('int64') 
    
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    # 80/20 split
    train_size = int((1 - test_split) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    return train_dataset, test_dataset