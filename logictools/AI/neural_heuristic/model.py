import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import OrderedDict
from functools import partial

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_gru_encoder(input_size, hidden_size, num_layers):
    return nn.GRU(input_size=input_size, hidden_size=hidden_size,
                  num_layers=num_layers, batch_first=True)


def get_transformer_encoder():
    pass


def get_mlp_decoder(input_size, output_size, output_activation,
                    layer_activation=nn.ReLU, scaledown_factor=2):
    layers, layer_count = [], 0
    while input_size // scaledown_factor > output_size:
        layers.append(
            (f"linear{layer_count}",
             nn.Linear(
                 input_size,
                 input_size //
                 scaledown_factor)))
        layers.append((f"activation{layer_count}", layer_activation()))
        input_size, layer_count = input_size // scaledown_factor, layer_count + 1
    layers.append((f"linear{layer_count}", nn.Linear(input_size, output_size)))
    layers.append((f"activation{layer_count}", output_activation()))

    return nn.Sequential(OrderedDict(layers))


class SiameseGruEncoderDecoderNetwork(nn.Module):

    def __init__(self, vocabulary_size, embedding_size, hidden_size, output_size, output_activation, encoder_layers=2,
                 decoder_activation=nn.ReLU, decoder_scaledown=2):
        super(SiameseGruEncoderDecoderNetwork, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.encoder_layers = encoder_layers

        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_size)
        self.encoder = get_gru_encoder(
            embedding_size, hidden_size, encoder_layers)
        self.decoder = get_mlp_decoder(hidden_size * 2, output_size, output_activation,
                                       layer_activation=decoder_activation, scaledown_factor=decoder_scaledown)

    def forward_one(self, x, x_lens, h):
        x = self.embedding_layer(x)
        xp = pack_padded_sequence(
            x,
            x_lens.cpu(),
            batch_first=True,
            enforce_sorted=False).to(device)
        out_packed, h = self.encoder(xp, h)
        out, out_len = pad_packed_sequence(out_packed, batch_first=True)
        return out, h

    def forward(self, X, h1, h2):
        w1, w2, l1, l2 = X
        if h1 is None:
            h1 = torch.randn(self.encoder_layers, w1.size(
                0), self.hidden_size).requires_grad_().to(device)
        if h2 is None:
            h2 = torch.randn(self.encoder_layers, w1.size(
                0), self.hidden_size).requires_grad_().to(device)

        w1_out, h1 = self.forward_one(w1, l1, h1)
        w2_out, h2 = self.forward_one(w2, l2, h2)
        # w1_out[:,-1,:], w2_out[:,-1,:] # only last hidden state
        w1_out, w2_out = w1_out.sum(dim=1), w2_out.sum(dim=1)

        join = torch.cat((w1_out, w2_out), dim=1)
        pred = self.decoder(join)
        return pred, h1, h2


class SiameseTransformerEncoderDecoderNetwork(nn.Module):
    pass


if __name__ == "__main__":
    print(device, "\n\n")

    sim_net_categorical = SiameseGruEncoderDecoderNetwork(
        10, 30, 60, 10, partial(nn.Softmax, dim=1))
    print(sim_net_categorical, "\n\n")

    sim_net_continuous = SiameseGruEncoderDecoderNetwork(10, 30, 60, 1, nn.ReLU, encoder_layers=3,
                                                         decoder_activation=nn.Sigmoid, decoder_scaledown=3)
    print(sim_net_continuous)
