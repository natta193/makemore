import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

# read names
words = open('data/names.txt').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)

block_size = 3 # how many characters to use to predict the next one

def build_dataset(words):
    # build the dataset
    X, Y = [], [] # inputs, labels (predictions)
    for w in words:
        #print(w)
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            # print(''.join(itos[i] for i in context), '--->', itos[ix])
            context = context[1:] + [ix] # crop and append

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

## build the different datasets ## 
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

# init
n_embd = 10 # demension of the character embeddings
n_hidden = 200 # number of hidden units in the hidden layer

# print(F.one_hot(torch.tensor(5), num_classes=27).float() @ C) # ALTERNATIVE

C = torch.randn((vocab_size, n_embd))
W1 = torch.randn((n_embd*block_size, n_hidden)) * (5/3)/(n_embd * block_size)**(5/3)/((n_embd * block_size)**0.5)
b1 = torch.randn(n_hidden) * 0.1
W2 = torch.randn((n_hidden, vocab_size)) * 0.1
b2 = torch.randn(vocab_size) * 0.1

bngain = torch.ones((1, n_hidden))*0.1 + 1.0
bnbias = torch.zeros((1, n_hidden))*0.1 # in charge of bias now

parameters = [C, W1, b1, W2, b2, bngain, bnbias] #, b1] # to train

print("Parameters:", sum(p.nelement() for p in parameters))

# hyperparameters
max_steps = 200000
batch_size = 32
n = batch_size
lossi = []

## TRAINING LOOP ##
with torch.no_grad():
    for iteration in (range(max_steps)):

        # minibatch construct
        ix = torch.randint(0, Xtr.shape[0], (batch_size,)) # better to have an approx gradient and make more steps then exact gradient and less steps
        Xb, Yb = Xtr[ix], Ytr[ix]

        # forward pass
        emb = C[Xb] # embed the characters into vectors 
        embcat = emb.view(emb.shape[0], -1) # concatenate the vectors
        # linear layer
        hprebn = embcat @ W1
        # batch norm layer
        bnmean = hprebn.mean(0, keepdim=True)
        bnvar = hprebn.var(0, keepdim=True, unbiased=True)
        bnvar_inv = (bnvar + 1e-5)**-0.5
        bnraw = (hprebn -bnmean) * bnvar_inv
        hpreact = bngain * bnraw + bnbias

        # non-linearity
        h = torch.tanh(hpreact) # hidden layer
        logits = h @ W2 + b2 # output layer
        loss = F.cross_entropy(logits, Yb) # loss function

        # backward pass
        dlogits = F.softmax(logits, 1)
        dlogits[range(n), Yb] -= 1
        dlogits /= n
        # 2nd layer backprop
        dh = dlogits @ W2.T
        dW2 = h.T @ dlogits
        db2 = dlogits.sum(0)
        # tanh
        dhpreact = (1.0 - h**2) * dh    
        # batchnorm backprop
        dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
        dbnbias = dhpreact.sum(0, keepdim=True)
        dhprebn = bngain*bnvar_inv/n * (n*dhpreact - dhpreact.sum(0) - n/(n-1)*bnraw*(dhpreact*bnraw).sum(0))
        # 1st layer
        dembcat = dhprebn @ W1.T
        dW1 = embcat.T @ dhprebn
        db1 = dhprebn.sum(0)
        # embedding
        demb = dembcat.view(emb.shape)
        dC = torch.zeros_like(C)
        for i in range(Xb.shape[0]):
            for j in range(Xb.shape[1]):
                ix = Xb[i, j]
                dC[ix] += demb[i, j]
        grads = [dC, dW1, db1, dW2, db2, dbngain, dbnbias]

        # update
        lr = 0.1 if i < 100000 else 0.01
        for p, grad in zip(parameters, grads):
            p.data += -lr * grad

        # track stats
        if iteration % 10000 == 0:
            print(f"{iteration:7d}/{max_steps:7d}: {loss.item():.4f}")
        lossi.append(loss.log10().item())

with torch.no_grad():
    emb = C[Xtr]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1 + b1
    bnmean = hpreact.mean(0, keepdim=True)
    bnvar = hpreact.var(0, keepdim=True, unbiased=True)

@torch.no_grad() # disables gradient tracking
def split_loss(split): # TEST
    x,y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]
    emb = C[x]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1 #+ b1
    # hpreact = bngain * (hpreact - hpreact.mean(0, keepdim=True)) / hpreact.std(0, keepdim=True) + bnbias # BATCH NORM
    hpreact = bngain * (hpreact - bnmean) / bnvar + bnbias # BATCH NORM
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

split_loss('train')
split_loss('val')

for _ in range(20):

    out = []
    context = [0] * block_size
    while True:
        # forward pass the neural net
        emb = C[torch.tensor([context])] # (1, block_size, n_embd)
        embcat = emb.view(emb.shape[0], -1)
        hpreact = embcat @ W1
        hpreact = bngain * (hpreact - bnmean) / bnvar + bnbias
        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        # sample from the distribution
        ix = torch.multinomial(probs[0], num_samples=1).item()
        # shift the context window and track the samples
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break

    print(''.join(itos[i] for i in out))

