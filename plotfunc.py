import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from snpmodel.dataset import amino_acid_alphabet
from snpmodel.mutant import all_possible_mutant_sequence
from snpmodel.train import predict


def featureMap_array(seq, model, aa_vocab, name="unk", fix_length=100):
    mutants_seqs, hgvs = all_possible_mutant_sequence(seq, fix_length)
    data = []
    for mutant_seq in mutants_seqs:
        m_tensor = torch.tensor(aa_vocab.vocab.lookup_indices(mutant_seq))
        m_tensor = F.one_hot(m_tensor, len(aa_vocab))
        m_tensor = m_tensor.float()
        data.append(m_tensor)
    dataIter = DataLoader(data, batch_size=512, shuffle=False)
    pred = predict(dataIter, model)
    hgvs = np.array(hgvs)
    return pred, hgvs

### 用于画图
def draw_mutant_heatmap(pred, name="unk", **kwargs):

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300, sharex=False)
    y_data = pred[:, :, 0]
    pic = sns.heatmap(y_data, ax = ax, xticklabels= list(amino_acid_alphabet.keys()),**kwargs)
    ax.set_title(f"{name}'s Benign or pathogenic map")




def qqplot(p_value:np.array):
    quantiles = np.linspace(0, 1, 100)  # 取100个分位数点出来作图

    observed = -np.log10(np.quantile(p_value, quantiles))

    expected = np.random.uniform(0, 1, 100000)
    expected = -np.log10(np.quantile(expected, quantiles))

    plt.scatter(x=expected, y=observed, color="black")
    plt.plot(expected, expected, linestyle="--", color="black", lw=1)
    plt.xlabel("Expected -log10 P-value")
    plt.ylabel("Observed -log10 P-value")
    plt.ylim(0, 10)
