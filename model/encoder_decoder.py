import torch
import torch.nn as nn
import random

# Encoder with lstm, rnn , gru 
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
        self.cell_type = cell_type.lower()

    def forward(self, src, src_lengths):
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths, batch_first=True, enforce_sorted=True)
        outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden

    def register_hooks(self):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                out_tensor = output[0]
            else:
                out_tensor = output
            # Skip if output is still a PackedSequence (from packed RNN)
            if isinstance(out_tensor, torch.nn.utils.rnn.PackedSequence):
                return
            self.saved_activations.append(out_tensor.detach().cpu())


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout, cell_type='gru'):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.cell_type = cell_type.lower()
        rnn_cls = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[self.cell_type]
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

        # LSTM returns (output, (h, c)), others return (output, h)
        if self.cell_type == 'lstm':
            rnn_output, (h, c) = self.rnn(embedded, hidden)
            hidden = (h, c)
        else:
            rnn_output, hidden = self.rnn(embedded, hidden)

        output = self.fc_out(rnn_output)  # (batch, 1, vocab)
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, max_length=25, sos_idx=1):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.max_length = max_length
        self.sos_idx = sos_idx

        enc_hidden_size = encoder.hidden_dim
        if encoder.bidirectional:
            assert decoder.rnn.hidden_size == enc_hidden_size * 2, \
                f"Decoder hidden dim must match encoder hidden dim x 2 when bidirectional. Got {decoder.rnn.hidden_size}, expected {enc_hidden_size * 2}"
        else:
            assert decoder.rnn.hidden_size == enc_hidden_size, \
                f"Decoder hidden dim must match encoder hidden dim. Got {decoder.rnn.hidden_size}, expected {enc_hidden_size}"

    def forward(self, src, tgt, src_lengths, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1) if tgt is not None else self.max_length
        output_dim = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, tgt_len, output_dim).to(self.device)

        encoder_outputs, hidden = self.encoder(src, src_lengths)
        if self.encoder.bidirectional:
            hidden = self._combine_bidirectional_hidden(hidden)

        if tgt is not None:
            input = tgt[:, 0]  # <sos>
        else:
            input = torch.tensor([self.sos_idx] * batch_size, dtype=torch.long, device=self.device)

        for t in range(1, tgt_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t, :] = output.squeeze(1)

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(2).squeeze(1)
            input = tgt[:, t] if (tgt is not None and teacher_force) else top1

        return outputs[:, 1:, :]

    def _combine_bidirectional_hidden(self, hidden):
        def reshape_and_concat(h):
            n_layers = self.encoder.n_layers
            batch_size = h.size(1)
            hidden_dim = h.size(2)
            h = h.view(n_layers, 2, batch_size, hidden_dim)
            return torch.cat((h[:, 0], h[:, 1]), dim=2)

        if self.encoder.cell_type == 'lstm':
            h, c = hidden
            return (reshape_and_concat(h), reshape_and_concat(c))
        else:
            return reshape_and_concat(hidden)


