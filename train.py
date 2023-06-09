from torch.utils.data import DataLoader
import model as md
import data_processing as data
import torch

def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_iter = data.Multi30k(split='train', language_pair=(data.SRC_LANGUAGE, data.TGT_LANGUAGE))
    train_dataloader = DataLoader(train_iter, batch_size=md.BATCH_SIZE, collate_fn=md.collate_fn)

    for src, tgt in train_dataloader:
        src = src.to(md.DEVICE)
        tgt = tgt.to(md.DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = md.create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = md.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return model, losses / len(list(train_dataloader))