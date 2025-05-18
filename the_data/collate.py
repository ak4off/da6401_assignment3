from torch.nn.utils.rnn import pad_sequence
import time

def collate_fn(batch):
    start = time.time()

    batch.sort(key=lambda x: len(x[0]), reverse=True)
    src_batch, tgt_batch = zip(*batch)
    src_lens = [len(seq) for seq in src_batch]
    tgt_lens = [len(seq) for seq in tgt_batch]

    src_padded = pad_sequence(src_batch, padding_value=0, batch_first=True)
    tgt_padded = pad_sequence(tgt_batch, padding_value=0, batch_first=True)

    # print(f"[DEBUG] Collate took {time.time() - start:.4f}s")
    return src_padded, tgt_padded, src_lens, tgt_lens
