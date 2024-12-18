import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

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

# hyperparameters
n_embd = 10 # demension of the character embeddings
n_hidden = 200 # number of hidden units in the hidden layer

# print(F.one_hot(torch.tensor(5), num_classes=27).float() @ C) # ALTERNATIVE

C = torch.randn((vocab_size, n_embd))
W1 = torch.randn((n_embd*block_size, n_hidden)) * (5/3)/(n_embd * block_size)**0.5 # 100 neurons - 30 because context of 3 , each is a 10d vector
                                                                                    # kaiming init - (5/3) for tanh
# b1 = torch.randn(n_hidden) * 0.01
W2 = torch.randn((n_hidden, vocab_size)) * 0.01
b2 = torch.randn(vocab_size) * 0

bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden)) # in charge of bias now
bnmean_running = torch.zeros((1, n_hidden))
bnstd_running = torch.ones((1, n_hidden))


parameters = [C, W1, W2, b2, bngain, bnbias] #, b1] # to train


print("Parameters:", sum(p.nelement() for p in parameters))

for p in parameters:
    p.requires_grad = True

# learn the learning rate
# lre = torch.linspace(-3, 0, 1000)
# lrs = 10**lre
# print(lrs)
# lri = []

# more hyperparameters
max_steps = 100000
batch_size = 64

lossi = []
# stepi = []

## TRAINING LOOP ##
for i in tqdm(range(max_steps)):

    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (batch_size,)) # better to have an approx gradient and make more steps then exact gradient and less steps
    Xb, Yb = Xtr[ix], Ytr[ix]

    # forward pass
    emb = C[Xb] # embed the characters into vectors 
    emb_cat = emb.view(-1, 30) # concatenate the vectors
    hpreact = emb_cat @ W1 #+ b1 # hidden layer pre-activation
    bnmeani = hpreact.mean(0, keepdim=True)
    bnstdi = hpreact.std(0, keepdim=True)
    hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias # BATCH NORM - introduces noise but helps with overfitting

    with torch.nograd():
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
        bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi

    h = torch.tanh(hpreact) # hidden layer
    logits = h@W2 + b2 # output layer
    loss = F.cross_entropy(logits, Yb) # loss function

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    # update
    lr = 0.1 if i < 50000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    # track stats
    # lri.append(lre[i])
    # stepi.append(i)
    if i % 10000 == 0:
        print(f"{i:7d}/{max_steps:7d}: {loss.item():.4f}")
    lossi.append(loss.log10().item())

# plt.figure(figsize=(20, 10))
# plt.imshow(h.abs() > 0.99, cmap='gray', interpolation='nearest')
# plt.show()

# plt.hist(h.view(-1).tolist(), 50)
# plt.show()

plt.plot(lossi) # plot the loss
plt.show()

## BATCH NORM CALIBRATION ## -- unneeded
# with torch.no_grad():
#     # pass the training set through
#     emb = C[Xtr]
#     embcat = emb.view(emb.shape[0], -1)
#     hpreact = embcat @ W1 + b1
#     # measure the mean/std over the entire training set
#     bnmean = hpreact.mean(0, keepdim=True)
#     bnstd = hpreact.std(0, keepdim=True)

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
    hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias # BATCH NORM
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
        h = torch.tanh(emb.view(1, -1) @ W1) #+ b1)
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
