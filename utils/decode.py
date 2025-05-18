def decode_sequence(output, vocab):
    # Decode sequences of token ids to actual strings
    return [vocab.ids2text(seq) for seq in output]


def get_char_level_confusion(preds, refs):
    from collections import Counter
    errors = Counter()
    for p, r in zip(preds, refs):
        for cp, cr in zip(p, r):
            if cp != cr:
                errors[(cr, cp)] += 1
    return errors.most_common(10)

# The function takes a tensor of token ids and a vocabulary object, and returns a list of decoded strings.
# The decoding process involves converting each token id back to its corresponding character using the vocabulary's `ids2text` method.
# The function is useful for interpreting the model's predictions in a human-readable format.
# The function is used in the `predict` function to convert the model's output into a format that can be saved or displayed.
# The function is also used in the `evaluate` function to decode the model's output for evaluation purposes.
# The function is defined in the `utils.decode` module, which is imported in the `predict` and `evaluate` modules.
# The function is not used in the `collate_fn` or `TransliterationDataset` classes, as those are responsible for preparing the data for training and evaluation.
# The function is not used in the `build_vocab` or `get_dataloaders` functions, as those are responsible for loading and preparing the data.
# The function is not used in the `train` or `evaluate` functions, as those are responsible for training and evaluating the model.