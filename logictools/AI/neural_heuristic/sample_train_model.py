from model import SiameseGruEncoderDecoderNetwork
from dataset import *
from train import *
import torch.nn as nn
import torch
from functools import partial

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sim_net_categorical_rule = SiameseGruEncoderDecoderNetwork(
    len(vocab_dict), 10, 30, len(rule_dict), partial(nn.Softmax, dim=1))
sim_net_categorical_step = SiameseGruEncoderDecoderNetwork(
    len(vocab_dict), 10, 30, 4, partial(nn.Softmax, dim=1))
sim_net_continuous_step = SiameseGruEncoderDecoderNetwork(
    len(vocab_dict), 10, 30, 1, nn.ReLU, encoder_layers=2, decoder_scaledown=3)

train_loader_rule, test_loader_rule = get_dataloader(
    RuleDataset, "training_datasets/rule_test_small.csv", 8)
train_loader_continuous_step, test_loader_continuous_step = get_dataloader(
    ContinuousStepDataset, "training_datasets/step_test_small.csv", 32, max_step_length=3)
train_loader_categorical_step, test_loader_categorical_step = get_dataloader(
    CategoricalStepDataset, "training_datasets/step_test_small.csv", 32, max_step_length=3)

if __name__ == "__main__":
    model_rule, losses_rule = train_rule_model(
        sim_net_categorical_rule, train_loader_rule, epochs=1)
    evaluate_accuracy(model_rule, test_loader_rule)
    print("\n\n")

    model_step_categorical, losses_step_categorical = train_categorical_step_model(
        sim_net_categorical_step, train_loader_categorical_step, epochs=1)
    evaluate_accuracy(model_step_categorical, test_loader_categorical_step)
    print("\n\n")

    model_step_continuous, losses_step_continuous = train_continuous_step_model(
        sim_net_continuous_step, train_loader_continuous_step, epochs=1)
    evaluate_continuous_data(
        model_step_continuous,
        test_loader_continuous_step)
