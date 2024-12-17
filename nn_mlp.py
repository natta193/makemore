import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import tqdm

# read names
words = open('data/names.txt').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

def build_dataset(words):
    # build the dataset
    block_size = 3 # how many characters to use to predict the next one
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

# print(F.one_hot(torch.tensor(5), num_classes=27).float() @ C) # ALTERNATIVE

C = torch.randn((27, 10))
W1 = torch.randn((30, 200)) # 100 neurons
b1 = torch.randn(200)
W2 = torch.randn((200, 27))
b2 = torch.randn(27)
parameters = [C, W1, b1, W2, b2]

print("Parameters:", sum(p.nelement() for p in parameters))

for p in parameters:
    p.requires_grad = True

# learn the learning rate
# lre = torch.linspace(-3, 0, 1000)
# lrs = 10**lre
# print(lrs)
# lri = []
lossi = []
stepi = []

for i in tqdm(range(200000)):
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (64,)) # better to have an approx gradient and make more steps then exact gradient and less steps

    # forward pass
    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    logits = h@W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    # update
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    # track stats
    # lri.append(lre[i])
    stepi.append(i)
    lossi.append(loss.log10().item())

emb = C[Xtr]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h@W2 + b2
loss = F.cross_entropy(logits, Ytr)

print(f"Training Loss: {loss.item()}")


emb = C[Xdev]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h@W2 + b2
loss = F.cross_entropy(logits, Ydev)

print(f"Validation Loss: {loss.item()}")

emb = C[Xte]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h@W2 + b2
loss = F.cross_entropy(logits, Yte)

print(f"Test Loss: {loss.item()}")

plt.plot(stepi, lossi) # plot the loss
plt.show()

# visualize embeddings
# plt.figure(figsize=(8,8))
# plt.scatter(C[:,0].data, C[:,1].data, s=200)
# for i in range(C.shape[0]):
#     plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha='center', color='white')
# plt.grid('minor')
# plt.show()
