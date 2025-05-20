# # wandb_sweep.py


import wandb
import os
import time 
current_time = time.strftime("%Y%m%d_%H%M%S")
sweep_config = {
    'method': 'bayes',  # Bayesian optimization for smarter sampling
    'name': f'A3_Translit_Sweep_both_{current_time}',
    'metric': {'name': 'val_loss', 'goal': 'minimize'},

    'parameters': {
        'in_embed_dims': {'values': [128, 256]},
        'n_layers': {'values': [1, 2]},
        'hidden_layer_size': {'values': [128, 256]},
        'dropout': {'values': [0.1, 0.3]},
        'learning_rate': {'values': [1e-4, 5e-4]},
        'batch_size': {'values': [32, 64]},
        'cell_type': {'values': ['rnn', 'lstm', 'gru']},
        'bidirectional': {'values': [True, False]},
        'weight_decay': {'values': [0.0, 1e-5]},
        'use_attention': {'values': [True, False]},  # set True if needed
        'n_epochs': {'values': [15, 30, 50]},
        # 'n_epochs': {'values': [1,2]},
        'max_length': {'values': [25]},
    }
}


def run_sweep():
    sweep_id = wandb.sweep(sweep_config, project="DA6401_A3_translit_hi", entity="ns24z274-iitm-ac-in")
    wandb.agent(sweep_id, function=launch_training, count=50)  


def launch_training():
    os.system("python Xtrain.py --use_wandb ")


if __name__ == "__main__":
    run_sweep()



