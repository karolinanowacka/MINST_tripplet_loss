class Trainer:
    def __init__(self, model, dataloader, optimizer, loss_fn, epochs = 10):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs

    def train(self):
        for epoch in range(self.epochs):
            for anchor_img, positive_img, negative_img in self.dataloader:
                self.optimizer.zero_grad()
                
                anchor_output = self.model(anchor_img)
                positive_output = self.model(positive_img)
                negative_output = self.model(negative_img)
                
                loss = self.loss_fn(anchor_output, positive_output, negative_output)
                loss.backward()
                self.optimizer.step()
                
            print(f'Epoch {epoch+1}, Batch [{i + 1}], Loss: {loss.item()}')