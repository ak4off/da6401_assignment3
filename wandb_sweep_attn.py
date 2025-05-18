# # wandb_sweep.py


import wandb
import os
import time 
current_time = time.strftime("%Y%m%d_%H%M%S")
sweep_config = {
    'method': 'bayes',  # Bayesian optimization for smarter sampling
    'name': f'Conne111Attention_{current_time}',
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
        'use_attention': {'values': [True]},  # set True if needed
        'n_epochs': {'values': [15, 30, 50]},
        # 'n_epochs': {'values': [1,2]},
        'max_length': {'values': [25]},
    }
}
#     'parameters': {
#         'in_embed_dims': {
#             'values': [64, 128, 256]  # Wider range to capture trade-offs
#         },
#         'n_layers': {
#             'values': [1, 2]  # Deeper layers rarely help for character-level tasks
#         },
#         'hidden_layer_size': {
#             'values': [128, 256, 512]  # Larger for attention models
#         },
#         'dropout': {
#             'values': [0.1, 0.2, 0.3]  # More granularity
#         },
#         'learning_rate': {
#             'distribution': 'log_uniform',  # Better for exponential scales
#             'min': 1e-5,
#             'max': 1e-3
#         },
#         'batch_size': {
#             'values': [32, 64]  # Keep if GPU memory is limited
#         },
#         'cell_type': {
#             'values': ['lstm', 'gru']  # Drop vanilla RNN (usually worse)
#         },
#         'bidirectional': {
#             'values': [True, False]
#         },
#         'weight_decay': {
#             'values': [0.0, 1e-5, 1e-4]  # Test stronger regularization
#         },
#         'use_attention': {
#             'values': [True]  # Force attention for Q5; disable for Q1-Q4
#         },
#         'n_epochs': {
#             'values': [20, 30]  # Early stopping will handle rest
#         },
#         'max_length': {
#             'values': [25]  # Fixed as per your data
#         }
#     },
#     'early_terminate': {  # Stop bad runs early
#         'type': 'hyperband',
#         'min_iter': 5,
#         'eta': 2
#     }
# }

def run_sweep():
    sweep_id = wandb.sweep(sweep_config, project="DA6401_A3_translit_hi", entity="ns24z274-iitm-ac-in")
    wandb.agent(sweep_id, function=launch_training, count=50)  


def launch_training():
    os.system("python Xtrain.py --use_wandb ")


if __name__ == "__main__":
    run_sweep()

# sweep_config = {
#     'method': 'bayes',
#     "name": "NEWSweep",

#     'metric': {'name': 'val_loss', 'goal': 'minimize'},
#     'parameters': {
#         'in_embed_dims': {'values': [128, 256]},
#         'n_layers': {'values': [1, 2]},
#         'hidden_layer_size': {'values': [128, 256]},
#         'dropout': {'values': [0.1, 0.3]},
#         'learning_rate': {'values': [1e-4, 5e-4]},
#         'batch_size': {'values': [32, 64]},
#         'cell_type': {'values': ['rnn', 'lstm', 'gru']},
#         'bidirectional': {'values': [True, False]},
#         'weight_decay': {'values': [0.0, 1e-5]},
#         'use_attention': {'values': [True, False]},  # set True if needed
#         'n_epochs': {'values': [15, 30, 50]},
#         # 'n_epochs': {'values': [1,2]},
#         'max_length': {'values': [25]},
#     }
# }

# gewep suggested
# sweep_config = {
#     'method': 'bayes',
#     'metric': {'name': 'val_acc', 'goal': 'maximize'},
#     'parameters': {
#         'embedding_dim': {'values': [16, 32, 64]},
#         'hidden_size': {'values': [64, 128, 256]},
#         'cell_type': {'values': ['rnn', 'gru', 'lstm']},
#         'dropout': {'values': [0.2, 0.3]},
#         'encoder_layers': {'values': [1, 2]},
#         'decoder_layers': {'values': [1, 2]},
#         'learning_rate': {'values': [0.001, 0.0005]},
#         'batch_size': {'values': [32, 64]},
#     }
# }


