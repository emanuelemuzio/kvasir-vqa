class EarlyStopper: 
    
    def __init__(self, patience=1, min_delta=0, min_epochs=0):
        self.min_epochs = min_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.epoch_counter = 0
        self.min_validation_loss = float('inf')
        
    def early_stop(self, validation_loss):
        self.epoch_counter += 1
        
        if self.epoch_counter < self.min_epochs:
            return False
        
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False