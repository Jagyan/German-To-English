from torch.utils.data import DataLoader
import model as md
import data_processing as data

def evaluate(model):
    model.eval()
    losses = 0

    val_iter = data.Multi30k(split='valid', language_pair=(data.SRC_LANGUAGE, data.TGT_LANGUAGE))
    val_dataloader = DataLoader(val_iter, batch_size=md.BATCH_SIZE, collate_fn=md.collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(md.DEVICE)
        tgt = tgt.to(md.DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = md.create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = md.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))