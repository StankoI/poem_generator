import torch

class LSTMLanguageModelPack(torch.nn.Module):
    def preparePaddedBatch(self, source):
        device = next(self.parameters()).device
        m = max(len(s) for (a,s) in source)
        sents = [[self.word2ind.get(w,self.unkTokenIdx) for w in s] for (a,s) in source]
        auths = [self.auth2id.get(a,0) for (a,s) in source]
        sents_padded = [ s+(m-len(s))*[self.padTokenIdx] for s in sents]
        return torch.t(torch.tensor(sents_padded, dtype=torch.long, device=device)), torch.tensor(auths, dtype=torch.long, device=device)
    
    def save(self,fileName):
        torch.save(self.state_dict(), fileName)
    
    def load(self,fileName,device):
        self.load_state_dict(torch.load(fileName,device))

    def __init__(self, embed_size, hidden_size, auth2id, word2ind, unkToken, padToken, endToken, lstm_layers, dropout): 
        super(LSTMLanguageModelPack, self).__init__()
        self.auth2id = auth2id
        self.word2ind = word2ind
        self.unkTokenIdx = word2ind[unkToken]
        self.padTokenIdx = word2ind[padToken]
        self.endTokenIdx = word2ind[endToken]
        self.lstm_layers = lstm_layers
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.embed = torch.nn.Embedding(len(word2ind), embed_size)
        self.authEmbed_h = torch.nn.Embedding(len(auth2id) + 1, lstm_layers * hidden_size)
        self.authEmbed_c = torch.nn.Embedding(len(auth2id) + 1, lstm_layers * hidden_size)
        self.lstm = torch.nn.LSTM(embed_size, hidden_size, lstm_layers, dropout=dropout)
        self.projection = torch.nn.Linear(hidden_size,len(word2ind))
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, source):    
        X, A = self.preparePaddedBatch(source)
        E = self.embed(X[:-1]) 
        source_lengths = [len(s) - 1 for (a, s) in source]
        
        batch_size = A.size(0)
        L = self.lstm_layers
        H = self.hidden_size
        
        h0 = self.authEmbed_h(A).view(batch_size, L, H).transpose(0, 1).contiguous()
        c0 = self.authEmbed_c(A).view(batch_size, L, H).transpose(0, 1).contiguous()
        
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            E, source_lengths, enforce_sorted=False
        )   
        
        outputPacked, _ = self.lstm(packed, (h0, c0))
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(outputPacked) 
        
        output = self.dropout(output)
        
        Z = self.projection(output.flatten(0, 1))  
        Y_bar = X[1:].flatten(0, 1)
        H = torch.nn.functional.cross_entropy(Z, Y_bar, ignore_index=self.padTokenIdx)
        
        return H
    

