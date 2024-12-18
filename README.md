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

- 200 Neurons 100000 cycles 10 embd 0.1 -> 50000 then 0.01 lr W2, B2 * 0.01, 0: - fixing softmax being confidently wrong
    - train 2.089956521987915
    - val 2.1519765853881836

- 200 Neurons 100000 cycles 10 embd 0.1 -> 50000 then 0.01 lr W2, B2, W1, B1 * 0.01, 0, 0.2, 0.01: - fixing tanh layer too saturated
    - train 2.043785810470581
    - val 2.106048107147217

- 200 Neurons 100000 cycles 10 embd 0.1 -> 50000 then 0.01 lr W2, B2, W1, B1 * 0.01, 0, 0.2, 0.01: - batch norm and kaiming init
    - train 2.067953109741211
    - val 2.126673698425293

- 200 Neurons 100000 cycles 10 embd 0.1 -> 50000 then 0.01 lr W2, B2, W1, B1 * 0.01, 0, 0.2, 0.01: - batch norm test optimization



BIGRAM:
gh. a. apsos. zh. ke. snde. h. mitedonthitlinyngene. chon. rilererete. ba. je. nela. zyeen. 
fhaatrolizikalomaivistauristriky. le. jhani. an.a.asedinadry.

TRIGRAM:
uskiberie. ous. xdyonne. oryenhur. hla. ul. uvielspofalrem. ede.
iayn. udaxizon. eiya. amrazmirienfhmycxtpe. uurazlermotobrx. alamredreemizeilondre. lieshan.
yr. yon. eashi. urgpori. ta.

MLP:
aedriseilahyah. palai. tricansh. araneydoyd. mahara. jah. nyn. nibton. colyn. ebzain.
koad. tanna. elioslynn. madessi. cerna. jakee. jomanxjonsonne. me. nayonit. prayah.

MLP2: (no batch norm)
terrine. dale. abduvann. ewnee. osue. mae. diz. reb. danceseriem. ayde. phiel. iseyli. ney.
rebel. ameki. kaylyn. keviannaelini. ersey. tai. dmaniaquan.
