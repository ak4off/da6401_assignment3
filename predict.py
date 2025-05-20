import torch
import os
from tqdm import tqdm
from utils.decode import decode_sequence
from utils.visualize import draw_connectivity, log_attention_to_wandb
import wandb
from utils.visualize import create_interactive_connectivity111

def pred_v_actual(input_vocab, output_vocab,src, tgt_out, predictions):
    table_pred = wandb.Table(columns=["Input", "Ground Truth", "Prediction"])

    # Inside evaluation loop (for N examples only)
    decoded_input = decode_sequence(src, input_vocab)
    decoded_target = decode_sequence(tgt_out, output_vocab)
    decoded_pred = decode_sequence(predictions, output_vocab)

    for i in range(min(5, len(decoded_input))):
        table_pred.add_data(decoded_input[i], decoded_target[i], decoded_pred[i])

    # After the loop
    wandb.log({"Prediction vs actual Table": table_pred})
def predict(model, dataloader, device, input_vocab, output_vocab):
    import os
    from tqdm import tqdm

    model.eval()
    predictions = []
    original_inputs = []
    ground_truths = []

    heatmap_table = wandb.Table(columns=["Input", "Prediction", "Attention Heatmap"])
    qn7_table = wandb.Table(columns=["Input", "Prediction", "INTERACTIVE Attention"])
    with torch.no_grad():
        # for batch_idx, (src, tgt_out, src_lengths, tgt_lengths) in enumerate(tqdm(dataloader, desc="Predicting")):
        for batch_idx, (src, tgt_out, src_lengths, tgt_lengths) in enumerate(dataloader):

            src = src.to(device)
            tgt_out = tgt_out.to(device)

            # Run model without teacher forcing
            output = model(src, None, src_lengths, teacher_forcing_ratio=0.0)
            output_logits = output[0]
            output_tokens = output_logits.argmax(dim=-1)

            decoded_output = decode_sequence(output_tokens, output_vocab)
            decoded_input = decode_sequence(src, input_vocab)
            decoded_gt = decode_sequence(tgt_out[:, 1:], output_vocab)     # skip <sos>

            predictions.extend(decoded_output)
            original_inputs.extend(decoded_input)
            ground_truths.extend(decoded_gt)
            table_pred1 = wandb.Table(columns=["Input", "Ground Truth", "Prediction"])
            wandb.log({"Prediction vs Ground Truth": table_pred1})

            # Compare predictions to ground truth (optional visual/metric logging)
            pred_v_actual(input_vocab, output_vocab, src.cpu(), tgt_out.cpu(), output_tokens.cpu())

            # If attention exists, visualize for first 3 examples
            if hasattr(model, "decoder") and hasattr(model.decoder, "attention"):
                attention_tensor = output[1]  # logits, attention

                for i in range(min(3, src.size(0))):
                    attn = attention_tensor[i].cpu()
                    inp_str = decoded_input[i]
                    pred_str = decoded_output[i]
                    src_tokens = list(inp_str)
                    tgt_tokens = list(pred_str)
                    # Static heatmap to wandb table
                    log_attention_to_wandb(attn, list(inp_str), list(pred_str), idx=i, table=heatmap_table)
                    # Generate interactive HTML visualization [tokens]
                    html_file = create_interactive_connectivity111(
                        attn_matrix=attn,
                        input_seq=src_tokens,
                        output_seq=tgt_tokens,
                        filename=f"attention_{batch_idx}_{i}.html"
                    )
                    # Log to wandb
                    wandb.log({
                        f"attention_html_{batch_idx}_{i}": wandb.Html(open(html_file).read())
                    })
                    # # Generate mapping HTML visualization
                    # html_filename = f"attention_{batch_idx}_{i}.html"
                    # create_interactive_connectivity111(
                    #     attn_matrix=attn,
                    #     input_seq=list(inp_str),
                    #     output_seq=list(pred_str),
                    #     filename=html_filename
                    # )

                    # # Log HTML to wandb
                    # with open(html_filename, 'r', encoding='utf-8') as f:
                    #     wandb.log({f"interactive_html_{batch_idx}_{i}": wandb.Html(f.read())})
                    qn7_table.add_data(
                        inp_str,
                        pred_str,
                        wandb.Html(open(html_file).read())  # Interactive visualization
                    )
                    os.remove(html_file)

    # Final wandb table log
    wandb.log({"Attention Table": heatmap_table})
    os.makedirs('predictions_attention', exist_ok=True)
    with open(f'predictions_attention/predictions_{wandb.run.id}.txt', 'w', encoding='utf-8') as f:
        for inp, pred, truth in zip(original_inputs, predictions, ground_truths):
            f.write(f"{inp}\t{pred}\t{truth}\n")

def predict_no_attention(model, dataloader, device, input_vocab, output_vocab):
    model.eval()
    predictions = []
    original_inputs = []
    # table = wandb.Table(columns=["Input", "Prediction"])

    with torch.no_grad():
        # for src, _, src_lengths, _ in tqdm(dataloader, desc="Predicting"):/
        # for src, tgt_inp, tgt_out, src_lengths, tgt_lengths in tqdm(dataloader, desc="Predicting"):
        # for src, tgt_out, src_lengths, tgt_lengths in tqdm(dataloader, desc="Predicting"):
        for src, tgt_out, src_lengths, tgt_lengths in dataloader:

            src = src.to(device)
            tgt_out = tgt_out.to(device)
            # No attention: model returns just logits
            output_logits = model(src, None, src_lengths, teacher_forcing_ratio=0.0)
            output_tokens = output_logits.argmax(dim=-1)

            decoded_output = decode_sequence(output_tokens, output_vocab)
            decoded_input = decode_sequence(src, input_vocab)

            predictions.extend(decoded_output)
            original_inputs.extend(decoded_input)
            # Log pred vs actual for this batch
            pred_v_actual(input_vocab, output_vocab, src.cpu(), tgt_out.cpu(), output_tokens.cpu())


