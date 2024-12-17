import torch
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F

words = open('data/names.txt').read().splitlines()
train = random.sample(words, int(0.8 * len(words))) # 80% train - 10% dev - 10% test
dev = random.sample(list(set(words) - set(train)), int(0.1 * len(words)))
test = list(set(words) - set(train) - set(dev))

# 2D array
chars = sorted(list(set(''.join(words))))

# conversion
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# create the training set of bigrams
xs, ys = [], [] # inputs, targets

# train set
for w in train:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()

# create the dev set of trigrams
xsd, ysd = [], [] # inputs, targets

for w in train:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xsd.append(ix1)
        ysd.append(ix2)
xsd = torch.tensor(xsd)
ysd = torch.tensor(ysd)
numd = xs.nelement()

# create the test set of trigrams
xst, yst = [], [] # inputs, targets

for w in train:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xst.append(ix1)
        yst.append(ix2)
xst = torch.tensor(xst)
yst = torch.tensor(yst)
numt = xs.nelement()

# init the network
W = torch.randn((27, 27), requires_grad=True) # generates 27 neurons weights for each of the 27 inputs
## if W is all equal, probabilities come out all uniform

## TRAINING LOOP ##
for _ in range(200):
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

    print(loss.item())

for i in range(20):
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

# evaluate on dev
t_loss = 0
num_loss = 0
for _ in range(200):
    ## FORWARD PASS ##
    xenc = F.one_hot(xsd, num_classes=27).float() # creates tensor with 0s and one 1 at xs' position
    logits = xenc @ W # log counts - logits ======= THE ONLY THING THAT CHANGES WHEN THE MODEL GETS MORE COMPLEX
    ## SOFTMAX start ## - aka normalization function
    counts = logits.exp() # counts - equivalent to the count matrix (but fake)
    probs = counts / counts.sum(1, keepdim=True)
    ## SOFTMAX end ##

    loss = -probs[torch.arange(numd), ysd].log().mean() # loss

    t_loss += loss.item()
    num_loss += 1

print(f"dev loss: {t_loss / num_loss}")

# evaluate on test
t_loss = 0
num_loss = 0
for _ in range(200):
    ## FORWARD PASS ##
    xenc = F.one_hot(xst, num_classes=27).float() # creates tensor with 0s and one 1 at xs' position
    logits = xenc @ W # log counts - logits ======= THE ONLY THING THAT CHANGES WHEN THE MODEL GETS MORE COMPLEX
    ## SOFTMAX start ## - aka normalization function
    counts = logits.exp() # counts - equivalent to the count matrix (but fake)
    probs = counts / counts.sum(1, keepdim=True)
    ## SOFTMAX end ##
    loss = -probs[torch.arange(numt), yst].log().mean() # loss

    t_loss += loss.item()
    num_loss += 1

print(f"test loss: {t_loss / num_loss}")
