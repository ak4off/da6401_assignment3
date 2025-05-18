import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout, bidirectional, cell_type='gru'):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.cell_type = cell_type.lower()
        rnn_cls = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[self.cell_type]
        self.rnn = rnn_cls(
            emb_dim,
            hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        # For activation tracking
        self.saved_activations = []
        self.save_activations = False

    def forward(self, src, src_lengths):
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths, batch_first=True, enforce_sorted=True)
        packed_outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        if self.save_activations:
            self.saved_activations.append(outputs.detach().cpu())
            self.saved_inputs.extend(seq for seq in src.detach().cpu())  # individual sequences

        return outputs, hidden

    def enable_activation_saving(self):
        """Call this once before training to enable PCA activation saving."""
        self.save_activations = True
        self.saved_activations = []  # Clear previous ones
        self.saved_inputs = []


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout, cell_type='gru'):
        super().__init__()
        self.cell_type = cell_type.lower()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        rnn_cls = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}[self.cell_type]
        self.rnn = rnn_cls(
            emb_dim,
            hidden_dim,
            n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        rnn_output, hidden = self.rnn(embedded, hidden)
        output = self.fc_out(rnn_output)
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, max_length=25, sos_idx=1):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.max_length = max_length
        self.sos_idx = sos_idx

        # Check decoder hidden compatibility
        enc_hidden_size = encoder.hidden_dim
        if encoder.bidirectional:
            assert decoder.rnn.hidden_size == enc_hidden_size * 2, \
                f"Expected decoder hidden size {enc_hidden_size * 2}, got {decoder.rnn.hidden_size}"
        else:
            assert decoder.rnn.hidden_size == enc_hidden_size, \
                f"Expected decoder hidden size {enc_hidden_size}, got {decoder.rnn.hidden_size}"

    def forward(self, src, tgt, src_lengths, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1) if tgt is not None else self.max_length
        output_dim = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, tgt_len, output_dim).to(self.device)

        encoder_outputs, hidden = self.encoder(src, src_lengths)

        hidden = self._combine_bidirectional_hidden(hidden) if self.encoder.bidirectional else hidden

        input = tgt[:, 0] if tgt is not None else torch.tensor([self.sos_idx] * batch_size, device=self.device)

        for t in range(1, tgt_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t, :] = output.squeeze(1)
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(2).squeeze(1)
            input = tgt[:, t] if teacher_force else top1

        return outputs[:, 1:, :]  # skip <sos>
    def _combine_bidirectional_hidden(self, hidden):
        def combine(h):
            n_layers = self.encoder.n_layers
            batch_size = h.size(1)
            hidden_dim = h.size(2)
            h = h.view(n_layers, 2, batch_size, hidden_dim)
            return torch.cat((h[:, 0, :, :], h[:, 1, :, :]), dim=2)

        if isinstance(hidden, tuple):
            h, c = hidden
            return (combine(h), combine(c))
        return combine(hidden)



class Attention(nn.Module):
    def __init__(self, decoder_hidden_dim, encoder_output_dim, attn_dim, device):
        super().__init__()
        self.device = device
        self.attn = nn.Linear(decoder_hidden_dim + encoder_output_dim, attn_dim)
        self.v = nn.Parameter(torch.rand(attn_dim))

    def forward(self, hidden, encoder_outputs):
        batch_size, seq_len, _ = encoder_outputs.size()
        attn_weights = torch.zeros(batch_size, seq_len).to(self.device)

        for t in range(seq_len):
            score = self.score(hidden, encoder_outputs[:, t])
            attn_weights[:, t] = score

        attn_weights = F.softmax(attn_weights, dim=1)
        self.last_attention_weights = attn_weights  # <--  Store for  use

        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        return context  # (B, 1, H_enc)

    def score(self, hidden, encoder_output):
        combined = torch.cat((hidden, encoder_output), dim=1)
        energy = torch.tanh(self.attn(combined))
        return torch.matmul(energy, self.v.unsqueeze(1)).squeeze(1)


class AttentionDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout, attention, encoder_output_dim, cell_type='gru'):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.attention = attention
        self.cell_type = cell_type.lower()
        rnn_cls = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}[self.cell_type]
        self.rnn = rnn_cls(
            emb_dim,
            hidden_dim,
            n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim + encoder_output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        rnn_output, hidden = self.rnn(embedded, hidden)
        # context = self.attention(hidden[-1], encoder_outputs)
        if isinstance(hidden, tuple):
            query = hidden[0][-1]  # use h_n
        else:
            query = hidden[-1]

        context = self.attention(query, encoder_outputs)

        output = torch.cat((rnn_output.squeeze(1), context.squeeze(1)), dim=1)
        output = self.fc_out(output).unsqueeze(1)
        #print(f"[Encoder] Returned hidden type: {type(hidden)}, LSTM = {'tuple' if isinstance(hidden, tuple) else 'tensor'}")

        return output, hidden

class Seq2SeqWithAttention(Seq2Seq):
    def __init__(self, encoder, decoder, device, max_length=25, sos_idx=1):
        super().__init__(encoder, decoder, device, max_length, sos_idx)
    def _is_lstm(self):
        return self.encoder.cell_type == 'lstm'

    def forward(self, src, tgt, src_lengths, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1) if tgt is not None else self.max_length
        output_dim = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, tgt_len, output_dim).to(self.device)
        all_attentions = []  # <-- NEW

        encoder_outputs, hidden = self.encoder(src, src_lengths)
        #print("\n[DEBUG] Encoder returned hidden state:")
        #print(f"  ➤ Type: {type(hidden)}")
        # if isinstance(hidden, tuple):
            ##print(f"  ➤ Tuple of shapes: {[h.shape for h in hidden]}")
        # else:
            ##print(f"  ➤ Shape: {hidden.shape}")
        hidden = self._combine_bidirectional_hidden(hidden) if self.encoder.bidirectional else hidden

        input = tgt[:, 0] if tgt is not None else torch.tensor([self.sos_idx] * batch_size, device=self.device)
        #print("[DEBUG] Passing hidden to decoder...")
        for t in range(1, tgt_len):
            #print(f"\n⏳ Time step {t}")
            #print(f"  ➤ Input shape: {input.shape}")
            #print(f"  ➤ Hidden type: {type(hidden)}")
            if isinstance(hidden, tuple):  # LSTM
                output, hidden = self.decoder(input, hidden, encoder_outputs)
                #print(f"  ➤ Hidden shapes: {[h.shape for h in hidden]}")
            else:  # GRU/RNN
                #print(f"  ➤ Hidden shape: {hidden.shape}")

                output, hidden = self.decoder(input, hidden, encoder_outputs)

            outputs[:, t, :] = output.squeeze(1)
        # Capture attention weights if the decoder stores them
            if hasattr(self.decoder.attention, 'last_attention_weights'):
                all_attentions.append(self.decoder.attention.last_attention_weights)
                
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(2).squeeze(1)
            input = tgt[:, t] if teacher_force else top1

        # Stack attention weights for visualization (B, T, S)
        if all_attentions:
            attention_tensor = torch.stack(all_attentions, dim=1)  # (B, tgt_len-1, src_len)
        else:
            attention_tensor = None
        return outputs[:, 1:, :], attention_tensor

        # return outputs[:, 1:, :]  # skip <sos>
