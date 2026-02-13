import torch

corpusFileName = 'corpusPoems'
modelFileName = 'modelLSTM.pt'
trainDataFileName = 'trainData'
testDataFileName = 'testData'
char2idFileName = 'char2id'
auth2idFileName = 'auth2id'

# device = torch.device("cuda:0")
device = torch.device("cpu")

batchSize = 48
char_emb_size = 48

hid_size = 384
lstm_layers = 4
dropout = 0.2

epochs = 5
learning_rate = 0.002

defaultTemperature = 0.3
