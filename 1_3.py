import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import dl.data
import dl.networks

torch.manual_seed(0)
#parameters
params = {
    'dim':64,
    'width':50,
    'depth':3,
    'lr':0.01,
    'epochs':25,
    'batch_size':10,
}
#using load digits data from dl.data
train_dataset,test_dataset = dl.data.load_digits_data()

train_loader = DataLoader(train_dataset,batch_size=params['batch_size'],shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=params['batch_size'],shuffle=False)
#numbers from 0 to 9 our output layer
in_dim = params['dim']
out_dim = 10
network = dl.networks.Dense(in_dim,out_dim,params['width'],params['depth'])

#cross entropy loss
optimizer = torch.optim.SGD(network.parameters(), lr=params['lr'])
loss_fn = F.cross_entropy

#25 epochs loop with backpropagation 
for epoch in range(params['epochs']):
    network.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = network(X_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{params['epochs']}, Loss: {total_loss / len(train_loader)}")

    
network.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_pred = network(X_batch)
        test_loss += loss_fn(y_pred, y_batch).item()
        pred = y_pred.argmax(dim=1,keepdim=True)
        correct += pred.eq(y_batch.view_as(pred)).sum().item()
#test loss and accuracy
test_loss /= len(test_loader)
accuracy = correct / len(test_loader.dataset)

print(f"\nTest Loss: {test_loss}, Test Accuracy: {accuracy * 100:.2f}%")

test_data = next(iter(test_loader))
X_test, y_test = test_data[0], test_data[1]
y_pred_test = network(X_test).argmax(dim=1)

fig,axes = plt.subplots(3,3,figsize=(8,8))
for i,ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8,8), cmap='gray')
    ax.set_title(f"True:{y_test[i]}, Pred:{y_pred_test[i]}")
    ax.axis('off')
    
#I saved a picture in folder
#plt.savefig('digit_predictions.png')

