import torch
import matplotlib.pyplot as plt

words = open('data/names.txt').read().splitlines()

# 2D array
chars = sorted(list(set(''.join(words))))

# conversion
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# create the training set of bigrams
xs, ys = [], [] # inputs, targets

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()

# init the network
import torch.nn.functional as F
W = torch.randn((27, 27), requires_grad=True) # generates 27 neurons weights for each of the 27 inputs
## if W is all equal, probabilities come out all uniform

## TRAINING LOOP ##
for _ in range(100):
    ## FORWARD PASS ##
    xenc = F.one_hot(xs, num_classes=27).float() # creates tensor with 0s and one 1 at xs' position
    logits = xenc @ W # log counts - logits ======= THE ONLY THING THAT CHANGES WHEN THE MODEL GETS MORE COMPLEX
    ## SOFTMAX start ## - aka normalization function
    counts = logits.exp() # counts - equivalent to the count matrix (but fake)
    probs = counts / counts.sum(1, keepdim=True)
    ## SOFTMAX end ##
    loss = -probs[torch.arange(num), ys].log().mean() # loss

    ## BACKWARD PASS ##
    W.grad = None # set to 0
    loss.backward() # backpropagation
    W.data += -50 * W.grad # update weights

for i in range(5):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdims=True)

        ix = torch.multinomial(p[0], num_samples=1, replacement=True).item()
        out.append(itos[ix])
        if ix == 0:
            break

    print(''.join(out))
