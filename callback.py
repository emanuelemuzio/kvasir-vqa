import logging
from datetime import datetime

now = datetime.now()
now = now.strftime("%Y-%m-%d")

logging.basicConfig(
    filename=f"logs/{now}.log",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    force=True,
    level=logging.INFO
)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            logging.info('Patience counter reset')
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            logging.info('Patience counter increased')
            self.counter += 1
            if self.counter >= self.patience:
                logging.info('Patience counter exceeded')
                return True
        return False