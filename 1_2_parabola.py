import torch
import torch.nn.functional as F

import dl
import dl.data
import dl.networks
import dl.models

torch.manual_seed(0)

#parameters
params = {
    'dim':2,  # input dimensions
    'width':50, # width of layers
    'depth':3,  # number of layers
    'lr':0.1,  # learning rate
    'epochs':25, # epochs for training
    'n_samples':100,  # training samples
    'batch_size':10,  # batch size
}

if __name__ == '__main__':
    
    #training data
    train_dataset = dl.data.parabola_nd(params['n_samples'], params['dim'])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params['batch_size'], shuffle=True)

    # input and output
    in_dim = train_dataset.data.shape[-1]  
    out_dim = train_dataset.targets.shape[-1]  
    network = dl.networks.Dense(in_dim,out_dim,params['width'],params['depth'])

    #optimizer SGD
    optimizer = torch.optim.SGD(network.parameters(), lr=params['lr'])
    model = dl.models.Model(network, optimizer, F.mse_loss, params['epochs'])

    #trainnig
    _,loss = model.fit(train_dataloader)

    #test data
    test_dataset = dl.data.parabola_nd(50, params['dim'])
    test_x = test_dataset.data
    test_y = test_dataset.targets


    test_pred = model.predict(test_x)

    #test loss
    test_loss = F.mse_loss(test_pred, test_y)
    
    print(f"Test loss: {test_loss.item()}")

