bigram:
- 1 smoothing: 2.4543561935424805
- 10 smoothing: 2.46164870262146

nn_bigram: 2.4621729850769043
- dev loss: 2.4629013538360596
- test loss: 2.4629013538360596

nn_trigram: 2.2483134269714355
- dev loss: 2.252880573272705
- test loss: 2.2606148719787598
- test loss with dev smoothing 0.01: 2.2604715824127197
- test loss with dev smoothing 0.1: 2.2825894355773926

nn_mlp:
- 100 Neurons 30000 cycles 2 embd 0.1 lr:
    - Training Loss: 2.4981648921966553
    - Validation Loss: 2.515202283859253

- 300 Neurons 30000 cycles 2 embd 0.1 lr:
    - Training Loss: 2.675812005996704
    - Validation Loss: 2.670238494873047

- 300 Neurons 100000 cycles 2 embd 0.1 lr:
    - Training Loss: 2.391284465789795
    - Validation Loss: 2.4003586769104004

- 200 Neurons 30000 cycles 10 embd 0.1 lr:
    - Training Loss: 2.3099002838134766
    - Validation Loss: 2.325669050216675

- 200 Neurons 60000 cycles 10 embd 0.05 lr:
    - Training Loss: 2.2728960514068604
    - Validation Loss: 2.2985517978668213

- 200 Neurons 200000 cycles 10 embd 0.1 -> 100000 then 0.01 lr:
    - Training Loss: 2.115107774734497
    - Validation Loss: 2.177528142929077
    - Test Loss: 2.173140287399292

BIGRAM:
gh.
a.
apsos.
zh.
ke.
snde.
h.
mitedonthitlinyngene.
chon.
rilererete.
ba.
je.
nela.
zyeen.
fhaatrolizikalomaivistauristriky.
le.
jhani.
an.
a.
asedinadry.

TRIGRAM:
uskiberie.
ous.
xdyonne.
oryenhur.
hla.
ul.
uvielspofalrem.
ede.
iayn.
udaxizon.
eiya.
amrazmirienfhmycxtpe.
uurazlermotobrx.
alamredreemizeilondre.
lieshan.
yr.
yon.
eashi.
urgpori.
ta.

MLP:
aedriseilahyah.
palai.
tricansh.
araneydoyd.
mahara.
jah.
nyn.
nibton.
colyn.
ebzain.
koad.
tanna.
elioslynn.
madessi.
cerna.
jakee.
jomanxjonsonne.
me.
nayonit.
prayah.


Exercises:
/ E01: train a trigram language model, i.e. take two characters as an input to predict the 3rd one. Feel free to use either counting or a neural net. Evaluate the loss; Did it improve over a bigram model?
/ E02: split up the dataset randomly into 80% train set, 10% dev set, 10% test set. Train the bigram and trigram models only on the training set. Evaluate them on dev and test splits. What can you see?
/ E03: use the dev set to tune the strength of smoothing (or regularization) for the trigram model - i.e. try many possibilities and see which one works best based on the dev set loss. What patterns can you see in the train and dev set loss as you tune this strength? Take the best setting of the smoothing and evaluate on the test set once and at the end. How good of a loss do you achieve?
E04: we saw that our 1-hot vectors merely select a row of W, so producing these vectors explicitly feels wasteful. Can you delete our use of F.one_hot in favor of simply indexing into rows of W?
E05: look up and use F.cross_entropy instead. You should achieve the same result. Can you think of why we'd prefer to use F.cross_entropy instead?
E06: meta-exercise! Think of a fun/interesting exercise and complete it.
