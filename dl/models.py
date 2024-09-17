class Model:

    def __init__(self, network, optimizer, loss_fn, epochs):

        self.network = network
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs

    def fit(self, dataloader):

        for epoch in range(self.epochs):
            for i, (x, y) in enumerate(dataloader):

                pred = self.network(x)
                loss = self.loss_fn(pred, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f'epoch {epoch}, loss {loss}')

        return self.network, loss

    def predict(self, *args, **kwargs):

        return self.network(*args, **kwargs)