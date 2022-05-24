import json
from random import random

import inspect
from Levenshtein import distance
import numpy as np

from logictools.AI.neural_embedding_heuristic import NeuralEmbeddingHeuristic


# all heuristics expect Tuple<expr: str, law:str> as inputs. Change typing to a StepNode : {expr: str, law: str} object

def random_weight(n1, n2):
    return random() * 10


def levenshtein_distance(n1, n2):
    return distance(n1[0], n2[0])


def len_distance(n1, n2):
    return abs(len(n1[0])-len(n2[0]))


def unitary_distance(n1, n2):
    return 1


def variable_mismatch(n1, n2):            # vars in n1 but not in n2 and vice versa
    cfunc = lambda x: 97 <= ord(x) <= 122 and x != 'v'
    n1v, n2v = set(filter(cfunc, n1[0])), set(filter(cfunc, n2[0]))
    return len((n1v | n2v) - (n1v & n2v))


class RuleDists:

    def __init__(self):
        self.all_dists = list(filter(lambda r: not r[0].startswith("__"), inspect.getmembers(self, predicate=inspect.ismethod)))
        self.all_dists = [r[1] for r in self.all_dists]

    def start_dist(self, n1, n2, d=1):
        return 0 if n1[1] == "Start" else d

    def iad_dist(self, n1, n2, d=1):
        return 0 if n1[1] == "Implication as Disjunction" else d

    def ifi_dist(self, n1, n2, d=1):
        return 0 if n1[1] == "Iff as Implication" else d

    def idemp_dist(self, n1, n2, d=1):
        return 0 if n1[1] == "Idempotence" else d

    def ident_dist(self, n1, n2, d=1):
        return 0 if n1[1] == "Identity" else d

    def dom_dist(self, n1, n2, d=1):
        return 0 if n1[1] == "Domination" else d

    def comm_dist(self, n1, n2, d=1):
        return 0 if n1[1] == "Commutativity" else d

    def assoc_dist(self, n1, n2, d=1):
        return 0 if n1[1] == "Associativity" else d

    def neg_dist(self, n1, n2, d=1):
        return 0 if n1[1] == "Negation" else d

    def absorb_dist(self, n1, n2, d=1):
        return 0 if n1[1] == "Absorption" else d

    def distr_dist(self, n1, n2, d=1):
        return 0 if n1[1] == "Distributivity" else d

    def demgn_dist(self, n1, n2, d=1):
        return 0 if n1[1] == "De Morgan's Law" else d


class MetaHeuristic:

    def __init__(self, heuristics=()):
        self.heuristics = heuristics
        self.weights = np.random.random(len(heuristics))

    def init_state(self, weight_file):
        self.heuristics, self.weights = [], []
        with open(weight_file, "r") as wf:
            for l in wf.readlines():
                heur, val = l.split(": ")
                if heur in globals():
                    self.heuristics.append(globals()[heur])
                else:
                    self.heuristics.append(getattr(RuleDists(), heur))
                self.weights.append(float(val))

    def set_weights(self, weights):
        self.weights = weights

    def meta_dist(self, n1, n2):
        ds = np.array([x(n1, n2) for x in self.heuristics])
        return np.sum(ds * self.weights)


class GeneHeuristic:

    def __init__(self, heuristics=None, weights=None, model_files=None):
        self.heuristics = heuristics
        self.weights = weights
        self.model_files = model_files
        self.params = {}

    def gene_meta_dist(self, n1, n2):
        ds = np.array([x(n1, n2) for x in self.heuristics])
        return np.sum(ds * self.weights)

    def set_params(self, params):
        self.params = params

    def load(self, weight_file):
        self.heuristics, self.weights = [], []
        with open(weight_file, "r") as wf:
            lines = list(wf.readlines())
            self.params = json.loads(lines[0].replace("'", "\""))
            for l in lines[2:]:
                try:
                    heur, val, file = l.split(": ")
                    self.heuristics.append(getattr(NeuralEmbeddingHeuristic(file[:-1], is_state_dict=True), heur))
                except ValueError:
                    heur, val = l.split(": ")
                    if heur in globals():
                        self.heuristics.append(globals()[heur])
                    else:
                        self.heuristics.append(getattr(RuleDists(), heur))
                self.weights.append(float(val))

    def save(self, out_file):
        with open(out_file, "w") as f:
            f.write(str(self.params)+"\n\n")
            for i, h in enumerate(self.heuristics):
                if self.model_files[i]:
                    f.write(f"{h.__name__}: {self.weights[i]}: {self.model_files[i]}\n")
                else:
                    f.write(f"{h.__name__}: {self.weights[i]}\n")


if __name__ == "__main__":
    n1, n2 = ('p->q', "Start"), ('p->q', None)
    gh = GeneHeuristic()
    gh.load("astar_heuristic_weights.txt")
    print(gh.gene_meta_dist(n1, n2))

