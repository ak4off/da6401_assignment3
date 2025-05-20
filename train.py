import os
os.environ['WANDB_INIT_TIMEOUT'] = '600'

import argparse
import time
import random
import numpy as np
import torch
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import io
import warnings
import wandb
from evaluate import evaluate
from predict import predict
from predict import predict_no_attention
from model.encoder_decoder import Seq2Seq, Encoder as VanillaEncoder, Decoder as VanillaDecoder
from model.attention_encoder_decoder import Attention, AttentionDecoder, Seq2SeqWithAttention, Encoder as AttnEncoder
from the_data.load_data import get_dataloaders
from utils.wandb_logger import setup_wandb
from utils.visualize import plot_attention
from evaluate import evaluate
import pickle
from torch.cuda.amp import autocast, GradScaler
from sklearn.decomposition import PCA
from utils.visualize import plot_grid_heatmaps, log_attention_table
import seaborn as sns

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def evaluate_and_visualize_no_attention(model, dataloader, criterion, device, input_vocab, output_vocab, epoch):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            # Adjust for no-attention: only 4 items
            src, tgt_inp, tgt_out, src_lens = batch
            src, tgt_inp, tgt_out = src.to(device), tgt_inp.to(device), tgt_out.to(device)

            # If your model does not expect tgt_lens, drop it
            output = model(src, tgt_inp, src_lens)  # [B, T, V]

            output_flat = output.view(-1, output.size(-1))   # [B*T, V]
            target_flat = tgt_out.view(-1)                   # [B*T]

            loss = criterion(output_flat, target_flat)
            total_loss += loss.item()

            predictions = output.argmax(dim=-1)  # [B, T]
            mask = tgt_out != output_vocab.pad_id
            correct = (predictions == tgt_out) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, accuracy


def evaluate_and_visualize(model, dev_loader, criterion, device, input_vocab, output_vocab, epoch=0,args=None):
    model.eval()
    total_loss = 0
    correct_tokens = 0
    total_tokens = 0
    attention_list = []
    input_list = []
    output_list = []
    MAX_ATTN_SAMPLES = 10

    with torch.no_grad():
        for batch_idx, (src, tgt, src_lengths, tgt_lengths) in enumerate(dev_loader):
            src, tgt = src.to(device), tgt.to(device)
            output, attention_weights, *_ = model(src, tgt, src_lengths)

            output_flat = output.reshape(-1, output.size(-1))
            target_flat = tgt[:, 1:].contiguous().view(-1)
            loss = criterion(output_flat, target_flat)
            total_loss += loss.item()

            pred_tokens = output.argmax(-1)
            tgt_tokens = tgt[:, 1:]
            mask = (tgt_tokens != output_vocab.stoi["<pad>"])
            correct_tokens += ((pred_tokens == tgt_tokens) & mask).sum().item()
            total_tokens += mask.sum().item()


           # Collect attention samples up to MAX_ATTN_SAMPLES
            if len(attention_list) < MAX_ATTN_SAMPLES:
                attention_list.append(attention_weights[0].cpu())
                # input_list.append([input_vocab.itos[i] for i in src[0]])
                # output_list.append([output_vocab.itos[i] for i in tgt[0]])
                src_len = src_lengths[0]
                tgt_len = attention_weights.shape[0]  # this should be the actual output length used

                input_list.append([input_vocab.itos[i] for i in src[0][:src_len]])
                output_list.append([output_vocab.itos[i] for i in tgt[0][1:1 + tgt_len]])
                
    avg_loss = total_loss / len(dev_loader)
    val_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0

    if wandb.run:
        wandb.log({
            "val_loss": avg_loss,
            "val_accuracy": val_acc
        })


        # Grid of 9 samples
        if len(attention_list) >= 9:
            plot_grid_heatmaps(attention_list[:9], input_list[:9], output_list[:9])

        # Table with 10 samples
        log_attention_table(attention_list[:10], input_list[:10], output_list[:10])

        # Log a single large attention heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(attention_list[0], xticklabels=input_list[0], yticklabels=output_list[0], cmap="viridis", ax=ax)
        ax.set_title(f"Attention Heatmap (Epoch {epoch})")
        wandb.log({"attention_heatmap": wandb.Image(fig)})
        plt.close(fig)

    return avg_loss, val_acc



def parse_args():
    parser = argparse.ArgumentParser(description="Training file for seq2seq model")
    parser.add_argument("-uw", "--use_wandb", action="store_true")
    parser.add_argument("-wp", "--wandb_project", type=str, default="DA6401 Assg3T1")
    parser.add_argument("-we", "--wandb_entity", type=str, default="ns24z274-iitm-ac-in")
    parser.add_argument("--run_name", type=str)
    parser.add_argument("-ie", "--in_embed_dims", type=int, default=256)
    parser.add_argument("-dl", "--n_layers", type=int, default=1)
    parser.add_argument("-hs", "--hidden_layer_size", type=int, default=256)
    parser.add_argument("-ct", "--cell_type", type=str, default="gru", choices=["rnn", "lstm", "gru"])
    parser.add_argument("-bi", "--bidirectional", action="store_true")
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate between 0 and 1')

    # parser.add_argument("-do", "--dropout", type=float, default=0.2)
    # parser.add_argument("-do", "--dropout", type=float, default=0.2, help="Dropout Rate")
    parser.add_argument("-ne", "--n_epochs", type=int, default=30)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    parser.add_argument("-ml", "--max_length", type=int, default=25)
    parser.add_argument("-bs", "--batch_size", type=int, default=32)
    parser.add_argument("--use_attention", action="store_true")
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def initialize_model(input_vocab_size, output_vocab_size, args, device, sos_idx):
    # Step 1: Initialize encoder
    encoder = AttnEncoder(  # or VanillaEncoder depending on your design
        input_dim=input_vocab_size,
        emb_dim=args.in_embed_dims,
        hidden_dim=args.hidden_layer_size,
        n_layers=args.n_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        cell_type=args.cell_type
    )

    # Step 2: Compute dimensions
    encoder_output_dim = args.hidden_layer_size * (2 if args.bidirectional else 1)
    decoder_hidden_dim = encoder_output_dim  # match dimensions

    # Step 3: Initialize attention if used
    if args.use_attention:
        attention = Attention(
            decoder_hidden_dim,
            encoder_output_dim,
            args.hidden_layer_size,
            device
        )
        decoder = AttentionDecoder(
            output_dim=output_vocab_size,
            emb_dim=args.in_embed_dims,
            hidden_dim=decoder_hidden_dim,
            encoder_output_dim=encoder_output_dim,
            n_layers=args.n_layers,
            dropout=args.dropout,
            attention=attention,
            cell_type=args.cell_type
        )
        model = Seq2SeqWithAttention(
            encoder=encoder,
            decoder=decoder,
            device=device,
            max_length=args.max_length,
            sos_idx=sos_idx
        )
    else:
        decoder = VanillaDecoder(
            output_dim=output_vocab_size,
            emb_dim=args.in_embed_dims,
            hidden_dim=decoder_hidden_dim,
            n_layers=args.n_layers,
            dropout=args.dropout,
            cell_type=args.cell_type
        )
        model = Seq2Seq(
            encoder=encoder,
            decoder=decoder,
            device=device,
            max_length=args.max_length,
            sos_idx=sos_idx
        )

    return model.to(device)



def main():
    wandb.init()
    args = parse_args()
    if wandb.run.sweep_id:
        sweep_config = wandb.config
        for key, value in sweep_config.items():
            setattr(args, key, value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    train_loader, dev_loader, test_loader, input_vocab, output_vocab = get_dataloaders(
        'data/train.tsv', 'data/dev.tsv', 'data/test.tsv', batch_size=args.batch_size, pin_memory=True)
    # # Save the vocabularies using pickle
    # with open('input_vocab.pkl', 'wb') as f:
    #     pickle.dump(input_vocab, f)

    # with open('output_vocab.pkl', 'wb') as f:
    #     pickle.dump(output_vocab, f)
    sos_idx = output_vocab.stoi["<sos>"]
    model = initialize_model(len(input_vocab), len(output_vocab), args, device, sos_idx) 
    model.encoder.enable_activation_saving()

    if args.use_wandb and not os.environ.get("WANDB_SWEEP"):
        setup_wandb(args.wandb_project, args.wandb_entity, model, run_name=args.run_name)
        wandb.config.update(vars(args))
        wandb.config.update({"seed": args.seed}, allow_val_change=True)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=output_vocab.stoi["<pad>"])

    best_loss = float('inf')
    patience = 5
    counter = 0
    scaler = GradScaler()
    start_time = time.time()

    for epoch in range(args.n_epochs):
        model.train()
        total_loss = 0
        correct_tokens = 0
        total_tokens = 0

        for src, tgt, src_lengths, tgt_lengths in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()

            with autocast(enabled=torch.cuda.is_available()):
                output = model(src, tgt, src_lengths) if not args.use_attention else model(src, tgt, src_lengths)[0]
                loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pred_tokens = output.argmax(-1)
            tgt_tokens = tgt[:, 1:]
            mask = (tgt_tokens != output_vocab.stoi["<pad>"])
            correct_tokens += ((pred_tokens == tgt_tokens) & mask).sum().item()
            total_tokens += mask.sum().item()

        avg_loss = total_loss / len(train_loader)
        train_acc = correct_tokens / total_tokens
        print(f"Epoch {epoch+1}/{args.n_epochs} - Loss: {avg_loss:.4f} - Accuracy: {train_acc:.4f}")

        if wandb.run:
            wandb.log({"epoch": epoch + 1,"train_loss": avg_loss, "train_accuracy": train_acc})

        # val_loss, val_acc = evaluate_and_visualize(model, dev_loader, criterion, device, input_vocab, output_vocab, epoch,args)
        # print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        if args.use_attention:
            val_loss, val_acc = evaluate_and_visualize(
                model, dev_loader, criterion, device, input_vocab, output_vocab, epoch, args)
        else:
            val_loss, val_acc = evaluate(model, dev_loader, criterion, device)
        #     val_loss, val_acc = evaluate_and_visualize_no_attention(
        #         model, dev_loader, criterion, device, input_vocab, output_vocab, epoch)

        scheduler.step(val_loss)

        if wandb.run:
            wandb.log({"val_loss": val_loss, "val_accuracy": val_acc, "lr": optimizer.param_groups[0]['lr']})

        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            best_model_state = model.state_dict()
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping after {epoch+1} epochs.")
                model.load_state_dict(best_model_state)
                break

    end_time = time.time()
    h, m = divmod(end_time - start_time, 3600)
    m, s = divmod(m, 60)
    print(f"Total training time: {int(h):02d}:{int(m):02d}:{int(s):02d}")

    # Evaluate on test set
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

    if wandb.run:
        wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})
    # os.makedirs("my_log", exist_ok=True)
    # save_path = f"my_log/best_model_{args.run_name or 'default'}.pt"
    # torch.save(best_model_state, save_path)
    # print(f"Model saved to {save_path}")
    # if wandb.run:
    #     artifact = wandb.Artifact("best_model", type="model")
    #     artifact.add_file(save_path)
    #     wandb.log_artifact(artifact)
    if args.use_attention:
        predict(model, test_loader, device, input_vocab, output_vocab)
    else:
        predict_no_attention(model, test_loader, device, input_vocab, output_vocab)


    # ========== PCA Visualization ==========
    if hasattr(model.encoder, 'saved_activations') and model.encoder.saved_activations:
        all_acts = torch.cat([a.reshape(-1, a.shape[-1]) for a in model.encoder.saved_activations], dim=0).cpu().numpy()
        all_inputs = torch.cat(model.encoder.saved_inputs).reshape(-1).cpu().numpy()
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(all_acts)

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=all_inputs, cmap='tab20', s=10)
        plt.title("PCA of Encoder Hidden States (colored by input tokens)")
        wandb.log({"encoder_pca": wandb.Image(fig)})
        plt.close()
    if wandb.run:
        wandb.finish()
if __name__ == "__main__":
    main()


# import os
# os.environ['WANDB_INIT_TIMEOUT'] = '600'
# import argparse
# import torch
# import torch.optim as optim
# import wandb
# import time
# from model.encoder_decoder import Seq2Seq, Encoder, Decoder
# from the_data.load_data import get_dataloaders
# from utils.wandb_logger import setup_wandb
# from evaluate import evaluate
# from predict import predict
# import random
# import numpy as np
# from model.attention_encoder_decoder import Attention,AttentionDecoder,Seq2SeqWithAttention # type: ignore
# import seaborn as sns
# from utils.visualize import plot_attention_heatmap, plot_grid_heatmaps,plot_attention  # Import visualization functions

# import warnings
# warnings.filterwarnings('ignore', category=UserWarning)  # Suppress user warnings
# warnings.filterwarnings('ignore', category=FutureWarning)  # Suppress future warnings


# import wandb
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image
# import io

# # def evaluate_and_visualize(model, dev_loader, criterion, device, output_vocab, idx=0):
# def evaluate_and_visualize(model, dev_loader, criterion, device, input_vocab, output_vocab, epoch=0):
#     model.eval()
#     total_loss = 0
#     correct_tokens = 0
#     total_tokens = 0
#     attention_list = []
#     input_list = []
#     output_list = []

#     with torch.no_grad():
#         for batch_idx, (src, tgt, src_lengths, tgt_lengths) in enumerate(dev_loader):
#             src, tgt = src.to(device), tgt.to(device)
#             output, attention_weights, *_ = model(src, tgt, src_lengths)
            
#             # Loss calculation
#             output_flat = output.reshape(-1, output.size(-1))
#             target_flat = tgt[:, 1:].contiguous().view(-1)
#             loss = criterion(output_flat, target_flat)
#             total_loss += loss.item()

#             # Accuracy calculation
#             pred_tokens = output.argmax(-1)
#             tgt_tokens = tgt[:, 1:]
#             mask = (tgt_tokens != output_vocab.stoi["<pad>"])
#             correct_tokens += ((pred_tokens == tgt_tokens) & mask).sum().item()
#             total_tokens += mask.sum().item()

#             # Collect attention data for visualization (first 3 batches only)
#             if batch_idx < 3:
#                 attention_list.append(attention_weights[0].cpu().numpy())
#                 input_list.append([input_vocab.itos[i] for i in src[0]])
#                 output_list.append([output_vocab.itos[i] for i in tgt[0]])

#     avg_loss = total_loss / len(dev_loader)
#     val_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0

#     # Visualization (log every 5 epochs)
#     if epoch % 5 == 0 and attention_list:
#         # Option 1: Use your existing plot_attention function
#         plot_attention(attention_weights[0], input_list[0], output_list[0], f"heatmap_epoch_{epoch}.png")
        
#         # Option 2: Direct W&B logging
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(attention_list[0], 
#                    xticklabels=input_list[0], 
#                    yticklabels=output_list[0], 
#                    cmap="viridis")
#         plt.title(f"Attention Heatmap (Epoch {epoch})")
#         plt.xlabel("Input (Latin)")
#         plt.ylabel("Output (Devanagari)")
        
#         wandb.log({
#             "attention_heatmap": wandb.Image(plt),
#             "val_loss": avg_loss,
#             "val_accuracy": val_acc
#         })
#         plt.close()

#     return avg_loss, val_acc
# # def evaluate_and_visualize(model, dev_loader, criterion, device, output_vocab, idx=0):
# #     model.eval()
# #     total_loss = 0
# #     correct_tokens = 0
# #     total_tokens = 0

# #     # Lists to store attention data for visualization
# #     attention_list = []
# #     input_list = []
# #     output_list = []

# #     with torch.no_grad():
# #         for src, tgt, src_lengths, tgt_lengths in dev_loader:
# #             src, tgt = src.to(device), tgt.to(device)
            
# #             # output, attention_weights = model(src, tgt, src_lengths)  # Modify the model to return attention weights
# #             output, attention_weights, *other_outputs = model(src, tgt, src_lengths)

# #             # # Flatten the output for the loss calculation
# #             # output_flat = output.view(-1, output.size(-1))  # (batch_size * seq_len, vocab_size)
# #             # target_flat = tgt[:, 1:].contiguous().view(-1)  # (batch_size * seq_len)

# #             # # Compute the loss
# #             # loss = criterion(output_flat, target_flat)
# #             # Reshape the output to match the shape of the target for cross-entropy loss
# #             output_flat = output.reshape(-1, output.size(-1))  # (batch_size * seq_len, vocab_size)
# #             target_flat = tgt[:, 1:].contiguous().view(-1)  # (batch_size * seq_len)
# #             assert output.shape[:-1] == tgt[:, 1:].shape, "Mismatch in output and target shapes before flattening"
# #             # Print shapes for debugging
# #             print(f"Output flattened shape: {output_flat.shape}")
# #             print(f"Target flattened shape: {target_flat.shape}")

# #             # Compute the loss
# #             loss = criterion(output_flat, target_flat)
# #             total_loss += loss.item()
# #             pred_tokens = output.argmax(-1)
# #             tgt_tokens = tgt[:, 1:]
# #             mask = (tgt_tokens != output_vocab.stoi["<pad>"])
# #             correct_tokens += ((pred_tokens == tgt_tokens) & mask).sum().item()
# #             total_tokens += mask.sum().item()

# #             # Collect attention data for visualization
# #             if idx < 3:  # Plot attention heatmaps for the first few examples
# #                 attention_list.append(attention_weights)  # Collect attention weights for visualization
# #                 input_list.append([output_vocab.itos[i] for i in src[0]])  # Convert indices to tokens for the input sequence
# #                 output_list.append([output_vocab.itos[i] for i in tgt[0]])  # Convert indices to tokens for the output sequence

# #             idx += 1

# #     avg_loss = total_loss / len(dev_loader)
# #     val_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0

# #     # Visualization
# #     if idx > 0:  # Only visualize if we have attention data
# #         plot_grid_heatmaps(attention_list, input_list, output_list)
    
# #     plot_attention(attention_weights[0], input_list[0], output_list[0], f"heatmap_{idx}.png")

# #     return avg_loss, val_acc




# def parse_args():
#     parser = argparse.ArgumentParser(description="Training file for seq2seq model")
#     parser.add_argument("-uw", "--use_wandb", action="store_true", help="Use Weights and Biases")
#     parser.add_argument("-wp", "--wandb_project", type=str, default="DA6401 Assg3T1", help="Project name for WandB")
#     parser.add_argument("-we", "--wandb_entity", type=str, default="ns24z274-iitm-ac-in", help="WandB entity name")
#     parser.add_argument("--run_name", type=str, default=None, help="Custom name for this wandb run")
#     parser.add_argument("-ie", "--in_embed_dims", type=int, default=256, help="Input Embedding Dimension")
#     parser.add_argument("-dl", "--n_layers", type=int, default=1, help="Number of Layers in Encoder and Decoder")
#     parser.add_argument("-hs", "--hidden_layer_size", type=int, default=256, help="Hidden Layer Size")
#     parser.add_argument("-ct", "--cell_type", type=str, default="gru", help="Cell Type: rnn, lstm, gru")
#     parser.add_argument("-bi", "--bidirectional", action="store_true", help="Use bidirectional RNNs")
#     parser.add_argument("-do", "--dropout", type=float, default=0.2, help="Dropout Rate")
#     parser.add_argument("-ne", "--n_epochs", type=int, default=30, help="Number of Epochs")
#     parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Learning Rate")
#     parser.add_argument("-ml", "--max_length", type=int, default=25, help="Max Sequence Length")
#     parser.add_argument("-bs", "--batch_size", type=int, default=32, help="Batch Size")
#     parser.add_argument("--use_attention", default=False, action="store_true", help="Use Attention mechanism")
#     parser.add_argument('--weight_decay', type=float, default=1e-5)
#     parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")


#     return parser.parse_args()


# # def initialize_model(input_vocab_size, output_vocab_size, args, device):
# def initialize_model(input_vocab_size, output_vocab_size, args, device, sos_idx):

#     output_dim = output_vocab_size
#     emb_dim = args.in_embed_dims
#     hidden_dim = args.hidden_layer_size
#     n_layers = args.n_layers
#     dropout = args.dropout


#     from model.attention_encoder_decoder import Encoder, Seq2SeqWithAttention

#     encoder = Encoder(input_vocab_size, args.in_embed_dims, args.hidden_layer_size,
#                       args.n_layers, args.dropout, args.bidirectional, cell_type=args.cell_type)



#     if args.use_attention:

#         from model.attention_encoder_decoder import Attention, AttentionDecoder, Seq2SeqWithAttention

#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         decoder_hidden_dim = hidden_dim * 2 if args.bidirectional else hidden_dim
#         encoder_output_dim = hidden_dim * 2 if args.bidirectional else hidden_dim

#         attention = Attention(
#             decoder_hidden_dim=decoder_hidden_dim,
#             encoder_output_dim=encoder_output_dim,
#             attn_dim=args.hidden_layer_size,  # or another hyperparameter
#             device=device
#         )


#         decoder = AttentionDecoder(
#             output_dim=output_vocab_size,
#             emb_dim=emb_dim,
#             hidden_dim=decoder_hidden_dim,
#             encoder_output_dim=encoder_output_dim,
#             n_layers=n_layers,
#             dropout=dropout,
#             attention=attention,
#             cell_type=args.cell_type
#         )

#         model = Seq2SeqWithAttention(
#             encoder=encoder,
#             decoder=decoder,
#             device=device,
#             max_length=args.max_length,
#             sos_idx=sos_idx
#         ).to(device)

#     else:
#         from model.encoder_decoder import Decoder

#         decoder_hidden_dim = hidden_dim * 2 if args.bidirectional else hidden_dim
#         decoder = Decoder(output_dim, emb_dim, decoder_hidden_dim, n_layers, dropout, cell_type=args.cell_type)

#         model = Seq2Seq(encoder, decoder, device, max_length=args.max_length, sos_idx=sos_idx).to(device)



#     return model


# def main():

#     wandb.init()
#     if wandb.run.sweep_id:
#         sweep_config = wandb.config
#         # First, parse the arguments normally
#         args = parse_args()
#         # Update the args with sweep_config values
#         for key, value in sweep_config.items():
#             setattr(args, key, value)
#     else:
#         args = parse_args()
#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Set seed: from args or default
#     seed = args.seed
#     # seed = 42
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
    
#     # Load data
#     train_loader, dev_loader, test_loader, input_vocab, output_vocab = get_dataloaders('data/train.tsv', 'data/dev.tsv', 'data/test.tsv', batch_size=args.batch_size,pin_memory=True)

#     # Initialize model
#     sos_idx = output_vocab.stoi["<sos>"]
#     model = initialize_model(len(input_vocab), len(output_vocab), args, device, sos_idx)
#     # model.encoder.register_hooks() # the 20m qn
#     model.encoder.enable_activation_saving()

#     # Setup wandb if enabled

#     if wandb.run and not os.environ.get("WANDB_SWEEP"):

#             setup_wandb(args.wandb_project, args.wandb_entity, model, run_name=args.run_name)
#             wandb.config.update(vars(args))
#             wandb.config.update({"seed": seed}, allow_val_change=True)

#     # print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")
#     # # Print model summary
#     # print(model)
#     # # Print vocab sizes
#     # print(f"Input vocab size: {len(input_vocab)}")
#     # print(f"Output vocab size: {len(output_vocab)}")
#     # # Print device
#     # print(f"Using device: {device}")
#     # # Print hyperparameters
#     # print(f"Hyperparameters: {args}")
#     # Optionally log hyperparameters to WandB   

#     # Define optimizer and loss function
#     # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
#     # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


#     criterion = torch.nn.CrossEntropyLoss(ignore_index=output_vocab.stoi["<pad>"])
#     # Scheduler to reduce LR on plateau
#     # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)


#     # Early stopping parameters
#     best_loss = float('inf')
#     patience = 5
#     counter = 0

#     start_time = time.time()

#     # Train the model
# #   AMP enabled training
#     from torch.cuda.amp import autocast, GradScaler

#     scaler = GradScaler()

#     for epoch in range(args.n_epochs):
#         model.train()
#         total_loss = 0
#         correct_tokens = 0
#         total_tokens = 0

#         for src, tgt, src_lengths, tgt_lengths in train_loader:
#             src, tgt = src.to(device), tgt.to(device)
#             optimizer.zero_grad()
#             use_amp = torch.cuda.is_available()

#             with autocast(enabled=use_amp):  # Mixed precision context
#                 if args.use_attention:
#                     output, _ = model(src, tgt, src_lengths)
#                 else:
#                     output = model(src, tgt, src_lengths)


#                 loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].contiguous().view(-1))


#             scaler.scale(loss).backward()
#             # Unscale the gradients before logging or stepping
#             scaler.unscale_(optimizer)

#             scaler.step(optimizer)
#             scaler.update()

#             total_loss += loss.item()

#             pred_tokens = output.argmax(-1)
#             tgt_tokens = tgt[:, 1:]
#             mask = (tgt_tokens != output_vocab.stoi["<pad>"])
#             correct_tokens += ((pred_tokens == tgt_tokens) & mask).sum().item()
#             total_tokens += mask.sum().item()

#         avg_loss = total_loss / len(train_loader)
#         train_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0
#         # Clear cache to avoid memory issues
#         torch.cuda.empty_cache()
#         train_acc_percent = train_acc * 100
#         avg_loss_percent = avg_loss * 100
#         print(f"Epoch {epoch+1}/{args.n_epochs} - Loss: {avg_loss_percent:.4f} - Accuracy: {train_acc_percent:.4f}")

#         if wandb.run:
#             wandb.log({"train_loss": avg_loss_percent, "train_accuracy": train_acc_percent})

#         val_loss, val_acc = evaluate(model, dev_loader, criterion, device)
#         # Inside the training loop, replace the validation evaluation with visualize code
#         # chcval_loss, checval_acc = evaluate_and_visualize(model, dev_loader, criterion, device, output_vocab) #give up

#         val_loss, val_acc = evaluate_and_visualize(model, dev_loader, criterion, device, input_vocab, output_vocab, epoch)
#         val_acc_percent = val_acc * 100
#         val_loss_percent = val_loss * 100
#         print(f"Validation Loss: {val_loss_percent:.4f}, Accuracy: {val_acc_percent:.4f}")
#         scheduler.step(val_loss)

#         if wandb.run:
#             wandb.log({"val_loss": val_loss_percent, "val_accuracy": val_acc_percent, "lr": optimizer.param_groups[0]['lr']})

#         if val_loss < best_loss:
#             best_loss = val_loss
#             counter = 0
#             best_model_state = model.state_dict()
#         else:
#             counter += 1
#             if counter >= patience:
#                 print(f"Early stopping triggered after {epoch+1} epochs.")
#                 model.load_state_dict(best_model_state)
#                 break


#     end_time = time.time()
#     elapsed_time = end_time - start_time

#     # Format and print total time
#     m, s = divmod(elapsed_time, 60)
#     h, m = divmod(m, 60)
#     print(f"Total training time: {int(h):02d}:{int(m):02d}:{int(s):02d}")
#     # Combine input character tensors
#     saved_input_tokens = model.encoder.saved_inputs

#     # # ======= Visualize Encoder Activations with PCA (Q6) =======
#     # if not hasattr(model.encoder, "saved_inputs"):
#     if hasattr(model.encoder, 'saved_activations') and model.encoder.saved_activations:

#         model.encoder.saved_inputs = []

#     model.encoder.saved_inputs.append(src.detach().cpu())

#     import matplotlib.pyplot as plt
#     from sklearn.decomposition import PCA

#     flat_activations = []
#     char_labels = []

#     for batch, src_batch in zip(model.encoder.saved_activations, saved_input_tokens):  # You'll define `saved_input_tokens` in a moment
#         # batch: (B, T, H), src_batch: (B, T)
#         B, T, H = batch.shape
#         flat_activations.append(batch.reshape(-1, H))
#         char_labels.extend(src_batch.reshape(-1).tolist())

#     flat_activations = torch.cat(flat_activations, dim=0)

#     # Run PCA
#     pca = PCA(n_components=2)
#     reduced = pca.fit_transform(flat_activations.numpy())

#     # Convert indices back to characters
#     char_strings = [input_vocab.itos[idx] if idx < len(input_vocab.itos) else '?' for idx in char_labels]
#     unique_chars = sorted(set(char_strings))
#     color_map = {char: i for i, char in enumerate(unique_chars)}
#     colors = [color_map[c] for c in char_strings]
#     # Plot with coloring by character
#     plt.figure(figsize=(10, 8))
#     # After generating the PCA plot (existing code)
#     # plt.figure(figsize=(10, 8))
#     sc = plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, cmap='tab20', alpha=0.5)
#     plt.colorbar(sc, ticks=range(len(unique_chars)), label="Character Index")
#     plt.title("PCA of Encoder Activations")

#     # Log to W&B
#     wandb.log({"pca_activations": wandb.Image(plt, caption="PCA of Encoder Hidden States")})
#     plt.close()
#     # unique_chars = sorted(set(char_strings))
#     # color_map = {char: i for i, char in enumerate(unique_chars)}
#     # colors = [color_map[c] for c in char_strings]

#     # sc = plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, cmap='tab20', alpha=0.5)
#     # plt.colorbar(sc, ticks=range(len(unique_chars)), label="Character Index")
#     # plt.clim(-0.5, len(unique_chars) - 0.5)
#     # plt.title("PCA of Encoder Activations Colored by Input Character")
#     # plt.xlabel("PC1")
#     # plt.ylabel("PC2")
#     # plt.tight_layout()
#     # plt.savefig("pca_colored_by_char.png")
#     # plt.show()


#     # Evaluate on test set
#     test_loss, test_acc = evaluate(model, test_loader, criterion, device)
#     print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

#     if wandb.run:
#         wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})

#     predict(model, test_loader, device, input_vocab, output_vocab)

#     if wandb.run:
#         wandb.finish()
# if __name__ == "__main__":
#     main()

