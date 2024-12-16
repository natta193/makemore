import torch
import matplotlib

words = open('data/names.txt').read().splitlines()

# 2D array 
chars = sorted(list(set(''.join(words))))

# conversion
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# create the training set of trigrams
xs, ys = [], [] # inputs, targets

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        ix3 = stoi[ch3]
        xs.append((ix1, ix2))
        ys.append(ix3)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.shape[0]

import torch.nn.functional as F
W = torch.randn((27 * 2, 27), requires_grad=True) # generates 27 neurons weights for each of the 27 inputs
# xenc = torch.Tensor()

## TRAINING LOOP ##
for _ in range(200):
    ## FORWARD PASS ##
    xenc = F.one_hot(xs, num_classes=27).float() # creates tensor with 0s and one 1 at xs' position
    xenc = xenc.view(num, -1)
    
    logits = xenc @ W

    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)

    loss = -probs[torch.arange(num), ys].log().mean() # loss

    ## BACKWARD PASS ##
    W.grad = None
    loss.backward()
    W.data += -50 * W.grad

    print(loss.item())

for i in range(5):
    out = []
    ix1, ix2 = 0, 0
    while True:
        xenc = F.one_hot(torch.tensor([ix1, ix2]), num_classes=27).float()
        xenc = xenc.view(1, -1)
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdim=True)

        ix3 = torch.multinomial(p[0], num_samples=1, replacement=True).item() # predict
        out.append(itos[ix3])
        if ix3 == 0:
            break

        ix1, ix2 = ix2, ix3

    print(''.join(out))
