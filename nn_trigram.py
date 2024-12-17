import torch
import matplotlib
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

# create the training set of trigrams
xs, ys = [], [] # inputs, targets

for w in train:
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

# create the dev set of trigrams
xsd, ysd = [], [] # inputs, targets

for w in dev:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        ix3 = stoi[ch3]
        xsd.append((ix1, ix2))
        ysd.append(ix3)
xsd = torch.tensor(xsd)
ysd = torch.tensor(ysd)
numd = xsd.shape[0]

# create the test set of trigrams
xst, yst = [], [] # inputs, targets

for w in test:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        ix3 = stoi[ch3]
        xst.append((ix1, ix2))
        yst.append(ix3)
xst = torch.tensor(xst)
yst = torch.tensor(yst)
numt = xst.shape[0]


# init the network
W = torch.randn((27 * 2, 27), requires_grad=True) # generates 27 neurons weights for each of the 27 inputs

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

print('----')

# evaluate on dev
for _ in range(200):
    ## FORWARD PASS ##
    xenc = F.one_hot(xsd, num_classes=27).float() # creates tensor with 0s and one 1 at xs' position
    xenc = xenc.view(numd, -1)
    
    logits = xenc @ W

    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)

    loss = -probs[torch.arange(numd), ysd].log().mean() + 0.1*(W**2).mean() # loss

    ## BACKWARD PASS ##
    W.grad = None
    loss.backward()
    W.data += -50 * W.grad

    print(loss.item())

# evaluate on test
t_loss = 0
num_loss = 0
for _ in range(200):
    ## FORWARD PASS ##
    xenc = F.one_hot(xst, num_classes=27).float() # creates tensor with 0s and one 1 at xs' position
    xenc = xenc.view(numt, -1)
    
    logits = xenc @ W

    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)

    loss = -probs[torch.arange(numt), yst].log().mean() # loss

    t_loss += loss.item()
    num_loss += 1

print(f"test loss: {t_loss / num_loss}")

for i in range(20):
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
