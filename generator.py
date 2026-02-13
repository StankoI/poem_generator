import numpy as np
import torch

def generateText(model, word2ind, auth, startSentence, limit=1500, temperature=1.0):
    startChar = '{'
    endChar   = '}'
    unkChar   = '@'

    # id -> token
    id2word = {i: w for w, i in word2ind.items()}

    device = next(model.parameters()).device
    model.eval()

    auth_id = model.auth2id.get(auth, 0)
    A = torch.tensor([auth_id], dtype=torch.long, device=device)  # (batch=1,)

    unk_idx = word2ind.get(unkChar, model.unkTokenIdx)
    end_idx = getattr(model, "endTokenIdx", None)

    # Начален резултат: обичайно махаме start token-а ако е в началото
    result = startSentence
    if result.startswith(startChar):
        result = result[1:]

    with torch.no_grad():
        # h0, c0 от автора (точно както във forward())
        L = model.lstm_layers
        H = model.hidden_size
        h = model.authEmbed_h(A).view(1, L, H).transpose(0, 1).contiguous()  # (L,1,H)
        c = model.authEmbed_c(A).view(1, L, H).transpose(0, 1).contiguous()  # (L,1,H)

        out = None

        # “нахрани” LSTM със seed-а
        for ch in startSentence:
            idx = word2ind.get(ch, unk_idx)
            x = torch.tensor([[idx]], dtype=torch.long, device=device)  # (seq=1,batch=1)
            e = model.embed(x)                                         # (1,1,E)
            out, (h, c) = model.lstm(e, (h, c))

        # генериране
        while len(result) < limit:
            last_h = out[-1, 0]                 # (hidden,)
            logits = model.projection(last_h)   # (vocab,)

            if temperature is None or temperature <= 0:
                next_idx = int(torch.argmax(logits).item())
            else:
                probs = torch.softmax(logits / float(temperature), dim=-1)
                p = probs.detach().cpu().numpy()
                next_idx = int(np.random.choice(len(p), p=p))

            if end_idx is not None and next_idx == end_idx:
                break

            next_ch = id2word.get(next_idx, unkChar)
            if next_ch == endChar:
                break

            result += next_ch

            x = torch.tensor([[next_idx]], dtype=torch.long, device=device)
            e = model.embed(x)
            out, (h, c) = model.lstm(e, (h, c))

    return result
