
import torch
from oodeel.methods import MLS, ODIN, DKNN, VIM, Energy, Entropy, Mahalanobis, Gram


Mls = MLS()
Odin = ODIN(temperature=1000)
Dknn = DKNN(nearest=50)
Vim = VIM(princ_dims=10)
energy = Energy()
entropy = Entropy()
mahalanobis = Mahalanobis(eps=0.002)
Gram = Gram(quantile=0.01)




