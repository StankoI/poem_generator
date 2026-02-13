import os
import numpy as np
import torch
import math

def trainModel(trainCorpus, lm, optimizer, epochs, batchSize,
               save_every=1, save_dir=None):

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    idx = np.arange(len(trainCorpus), dtype='int32')
    lm.train()

    for epoch in range(epochs):
        np.random.shuffle(idx)

        for b in range(0, len(idx), batchSize):
            batch = [trainCorpus[i] for i in idx[b:min(b+batchSize, len(idx))]]
            H = lm(batch)
            optimizer.zero_grad()
            H.backward()
            optimizer.step()

            print("Epoch:",epoch,'/',epochs,", Batch:",b // batchSize, '/', len(idx) // batchSize, ", loss: ",H.item())

        if save_dir is not None and (epoch + 1) % save_every == 0:
            save_path = os.path.join(save_dir, f"model_epoch{epoch+1}.pt")
            lm.save(save_path)
            print(f"Model saved to {save_path}")


def perplexity(lm, testCorpus, batchSize):
    lm.eval()
    H = 0.
    c = 0
    for b in range(0,len(testCorpus),batchSize):
        batch = testCorpus[b:min(b+batchSize, len(testCorpus))]
        l = sum(len(s)-1 for s in batch)
        c += l
        with torch.no_grad():
            H += l * lm(batch)
    return math.exp(H/c)
