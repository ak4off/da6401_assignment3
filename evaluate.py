import torch
from torch.utils.data import DataLoader
from utils.decode import decode_sequence

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        # for src, tgt, _, _ in dataloader:
        for src, tgt, src_lengths, _ in dataloader:

            src, tgt = src.to(device), tgt.to(device)
            if hasattr(model, "decoder") and hasattr(model.decoder, "attention"):
                output, _ = model(src, tgt, src_lengths)
            else:
                output = model(src, tgt, src_lengths)
            # # simpler versiponm
            # output_tuple = model(src, tgt, src_lengths)
            # output = output_tuple[0] if isinstance(output_tuple, tuple) else output_tuple

            # output = model(src, tgt, src_lengths, teacher_forcing_ratio=0.0)
            # output = model(src, tgt, teacher_forcing_ratio=0.0)

            # Compute loss
            # loss = criterion(output.view(-1, output.size(-1)), tgt[:, 1:].contiguous().view(-1))
            loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].contiguous().view(-1))

            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(output, dim=-1)
            correct += (predicted == tgt[:, 1:]).sum().item()
            total += tgt[:, 1:].numel()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy
