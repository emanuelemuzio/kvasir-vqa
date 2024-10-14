import matplotlib.pyplot as plt
import json, os
from dotenv import load_dotenv

load_dotenv()

def plot_run(base_path, run_id):
    run_path = f"{base_path}/{run_id}/run.json"
    if os.path.exists(run_path):
        with open(run_path, 'r') as file:
            data = json.load(file)

            actual_epochs = list(range(1, len(data['train_loss']) + 1))

            plt.plot(actual_epochs, data['train_loss'], 'r', label="Train loss")
            plt.plot(actual_epochs, data['val_loss'], 'g', label="Val loss")
            plt.plot(actual_epochs, data['val_acc'], 'b', label="Val acc")
            plt.legend(loc="upper right")

            plt.savefig(f"{base_path}/{run_id}/run.png")