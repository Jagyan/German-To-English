######################################################################
# Now we have all the ingredients to train our model. Let's do it!
#

from timeit import default_timer as timer
import model as md
import data_processing as data
import eval
import train
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default = "translate_1.pth", help = "Path to the model")
parser.add_argument("--mode", default = "evaluate", help = "Mode of operation: train/evaluate/transfer")
parser.add_argument("--phrase", default = "Herzlich Willkommen!", help = "German phrase to translate")
parser.add_argument("--num_epoch", default = 18, type = int, help = "Number of epochs to train")

def train_model(model, start = 1, epochs = 18):
    end = start + epochs
    for epoch in range(start, end):
        start_time = timer()
        trained_model, train_loss = train.train_epoch(model, md.optimizer)
        torch.save(trained_model, "models/translate_" + str(epoch) + ".pth")
        end_time = timer()
        val_loss = eval.evaluate(trained_model)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

    return trained_model


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(md.DEVICE)
    src_mask = src_mask.to(md.DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(md.DEVICE)
    for i in range(max_len-1):
        memory = memory.to(md.DEVICE)
        tgt_mask = (md.generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(md.DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == data.EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = md.text_transform[data.SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=data.BOS_IDX).flatten()
    return " ".join(data.vocab_transform[data.TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")


######################################################################
#
if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    if(args.mode == "train"):
        loaded_model = train_model(md.transformer, epochs = args.num_epoch)
    elif(args.mode == "translate"):
        loaded_model = torch.load("models/" + args.model_path)
        start = int(args.model_path.split("_")[1].split(".")[0]) + 1
        train_model(loaded_model, start = start, epochs = start+args.num_epoch)
    else:
        loaded_model = torch.load("models/" + args.model_path)

    
    print(translate(loaded_model, args.phrase))