from torch.utils.data import Dataset, DataLoader, random_split
from torch import Generator
import torch
from torch.nn.utils.rnn import pad_sequence


vocab_dict = {chr(i + 97): i + 1 for i in range(26)}
vocab_dict.update({k: i + 27 for i, k in enumerate(
    ['V', '^', 'T', 'F', '(', ')', '~', '-', '<', '>', '=']
)})
vocab_dict["<PAD>"] = 0
reverse_vocab = {v: k for k, v in vocab_dict.items()}

rule_dict = {r: i for i, r in enumerate([
    "Double Negation", "Implication as Disjunction", "Iff as Implication",
    "Idempotence", "Identity", "Domination", "Commutativity", "Associativity",
    "Negation", "Absorption", "Distributivity", "De Morgan's Law"
])}
reverse_rule = {v: k for k, v in rule_dict.items()}


class LogicLearnerBaseDataset(Dataset):

    def __init__(self, data_file):
        super(LogicLearnerBaseDataset).__init__()
        self.data_file = data_file

        with open(self.data_file, "r") as df:  # done in memory
            self.data = df.readlines()
            self.len = len(self.data)

    def __getitem__(self, item):
        raise NotImplementedError("Extend or use a subclass of this dataset.")

    def __len__(self):
        return self.len


class RuleDataset(LogicLearnerBaseDataset):

    def __init__(self, data_file):
        super(RuleDataset, self).__init__(data_file)

    def __getitem__(self, idx):
        def get_tensor(expr):
            return torch.tensor([vocab_dict[i] for i in list(expr)])

        item = self.data[idx][:-1]
        w1, w2, rule = item.split(",")
        return get_tensor(w1), get_tensor(w2), rule_dict[rule]


class CategoricalStepDataset(LogicLearnerBaseDataset):

    def __init__(self, data_file, max_step_length):
        super(CategoricalStepDataset, self).__init__(data_file)
        self.max_step_length = max_step_length

    def __getitem__(self, idx):
        def get_tensor(expr):
            return torch.tensor([vocab_dict[i] for i in list(expr)])

        item = self.data[idx][:-1]
        w1, w2, steps = item.split(",")
        return get_tensor(w1), get_tensor(w2), min(
            int(steps), self.max_step_length)


class ContinuousStepDataset(LogicLearnerBaseDataset):

    def __init__(self, data_file, max_step_length):
        super(ContinuousStepDataset, self).__init__(data_file)
        self.max_step_length = max_step_length

    def __getitem__(self, idx):
        def get_tensor(expr):
            return torch.tensor([vocab_dict[i] for i in list(expr)])

        item = self.data[idx][:-1]
        w1, w2, steps = item.split(",")
        return get_tensor(w1), get_tensor(w2), min(
            int(steps), self.max_step_length) / float(self.max_step_length)


def pad_collate(batch):
    w1, w2, sim = zip(*batch)
    l1, l2 = torch.tensor([len(x)
                          for x in w1]), torch.tensor([len(x) for x in w2])
    w1 = pad_sequence(w1, batch_first=True, padding_value=0)
    w2 = pad_sequence(w2, batch_first=True, padding_value=0)

    return w1, w2, sim, l1, l2


def get_dataloader(dataset_class, data_file, batch_size,
                   train_frac=0.7, random_seed=42, **kwargs):
    sd = dataset_class(data_file, **kwargs)
    split_len = int(len(sd) * train_frac)
    train_data, test_data = random_split(
        sd, (split_len, len(sd) - split_len), generator=Generator().manual_seed(random_seed)
    )
    train_loader = DataLoader(
        train_data, batch_size, collate_fn=pad_collate, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, collate_fn=pad_collate)
    return train_loader, test_loader


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
