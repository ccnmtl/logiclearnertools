import torch.nn as nn
from torch import optim, Generator
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader, random_split
from logictools.AI.sim_net import SimNet, NormDistNet

import os
from time import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NeuralEmbeddingHeuristic:

    def __init__(self, model_file, is_state_dict=True):
        self.model_file = model_file

        self.vocab_dict = {chr(i + 97): i + 1 for i in range(26)}
        self.vocab_dict.update({k: i + 27 for i, k in enumerate(
            ['V', '^', 'T', 'F', '(', ')', '~', '-', '<', '>', '=']
        )})
        self.vocab_dict["<PAD>"] = 0
        self.reverse_vocab = {v: k for k, v in self.vocab_dict.items()}

        if is_state_dict:
            self.model = NormDistNet(vocab_size=len(self.vocab_dict)) #SimNet(vocab_size=len(self.vocab_dict))
            self.model.load_state_dict(torch.load(model_file, map_location=device))
        else:
            self.model = torch.load(model_file, map_location=device)

        self.param_dict = {k: v for k, v in self.model.named_parameters()}
        # print({k: v.shape for k, v in self.param_dict.items()})
        self.model.eval()



    def expr_to_vocab(self, expr):
        return torch.tensor([self.vocab_dict[i] for i in list(expr)])

    def sim_to_dist(self, sim):
        '''sim = torch.squeeze(sim)
        #print(sim, torch.argmax(sim))
        return torch.argmax(sim).item() # 1 / (sim + 1e-8)'''
        return sim.item()

    def embedding_dist(self, n1, n2):
        e1, e2 = self.expr_to_vocab(n1[0]), self.expr_to_vocab(n2[0])
        # print(e1, e2)
        l1, l2 = torch.tensor([len(e1)]), torch.tensor([len(e2)])
        e1, e2 = e1[None, :], e2[None, :]
        # print(e1, e2, l1, l2)

        h1 = torch.zeros((self.model.layer_dim, e1.size(1), self.model.hidden_dim)).requires_grad_().to(device)
        h2 = torch.zeros((self.model.layer_dim, e1.size(1), self.model.hidden_dim)).requires_grad_().to(device)
        with torch.no_grad():
            sim, _, _ = self.model.forward((e1, e2, l1, l2), h1, h2)
        # print(f"Sim: {sim.item()}")
        return self.sim_to_dist(sim)


if __name__ == "__main__":
    n1, n2 = ('p', "Start"), ('p', None)
    n3, n4 = ('p->q', "Start"), ('p->qvr', None)
    n5, n6 = ('p->q', "Start"), ('~pvq', None)
    n7, n8 = ('p->q', "Start"), ('~q^~q^r^q', None)

    neh = NeuralEmbeddingHeuristic(os.path.join("models", "sol_dist_biased_model_parameters.pt"))
    print(n1[0], n2[0], neh.embedding_dist(n1, n2))
    print(n3[0], n4[0], neh.embedding_dist(n3, n4))
    print(n5[0], n6[0], neh.embedding_dist(n5, n6))
    print(n7[0], n8[0], neh.embedding_dist(n7, n8))
