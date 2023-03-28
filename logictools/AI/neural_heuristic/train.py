import torch.nn as nn
import torch
import sklearn.metrics as metrics

from dataset import reverse_rule

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def print_training_result(target_type, current_epoch,
                          total_epochs, batch_num, loss, sample):
    print(
        f"Epoch {current_epoch}/{total_epochs} Batch {batch_num}:\tLoss {loss}")
    if target_type.lower() == "rule":
        print(
            f"Output: {reverse_rule[torch.argmax(sample[0]).item()]}\tTarget: {reverse_rule[sample[1].item()]}")
    else:
        print(
            f"Output: {torch.argmax(sample[0]).item()}\tTarget: {sample[1].item()}")


def train_sequence_model(model, train_loader, target_type,
                         loss_func, epochs=5, lr=0.1, print_frequency=50):
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_batches = len(list(train_loader))
    train_losses, lowest_loss = [], float("inf")
    for ep in range(epochs):
        epoch_loss = 0
        for batch_num, (w1, w2, target, l1, l2) in enumerate(train_loader):
            target = torch.tensor(target).to(device)
            inputs = [
                w1.to(device),
                w2.to(device),
                l1.to(device),
                l2.to(device)]

            optimizer.zero_grad()
            h1, h2 = None, None
            out, h1, h2 = model(inputs, h1, h2)
            loss = loss_func(torch.squeeze(out), target)
            loss.backward()
            optimizer.step()

            if batch_num % print_frequency == 0:
                print_training_result(
                    target_type, ep + 1, epochs, batch_num, loss, (out[0], target[0]))

            epoch_loss += loss.cpu().detach()

        epoch_loss /= num_batches
        train_losses.append(epoch_loss)
        print("Epoch {}:\tAverage Loss: {}\n".format(ep + 1, epoch_loss))

        if epoch_loss < lowest_loss - 0.1:
            lowest_loss = epoch_loss
            torch.save(model, f"{target_type}-best-model.pt")
            torch.save(
                model.state_dict(),
                f"{target_type}-best-model-parameters.pt")
            print("Saved!")

    return model.cpu(), train_losses


def train_rule_model(model, train_loader, epochs=5,
                     lr=0.1, print_frequency=50):
    return train_sequence_model(
        model, train_loader, "rule", nn.CrossEntropyLoss(), epochs, lr, print_frequency)


def train_categorical_step_model(
        model, train_loader, epochs=5, lr=0.1, print_frequency=50):
    return train_sequence_model(model, train_loader, "categorical_step",
                                nn.CrossEntropyLoss(), epochs, lr, print_frequency)


def train_continuous_step_model(
        model, train_loader, epochs=5, lr=0.1, print_frequency=50):
    return train_sequence_model(
        model, train_loader, "continuous_step", nn.MSELoss(), epochs, lr, print_frequency)


def evaluate_accuracy(model, test_loader):
    correct, total = 0, 0
    model.eval()
    model.to(device)
    with torch.no_grad():
        for w1, w2, target, l1, l2 in test_loader:
            target = torch.tensor(target).to(device)
            inputs = [
                w1.to(device),
                w2.to(device),
                l1.to(device),
                l2.to(device)]

            out, _, _ = model(inputs, None, None)
            _, pred = torch.max(out.data, 1)
            correct += (pred == target).sum().item()
            total += len(target)
    print(f"Evaluated {total} - Accuracy: {correct * 100 / total}%")


def evaluate_continuous_data(model, test_loader):
    total = 0
    n_batches = len(test_loader)
    total_mse, total_mae, total_r2 = 0, 0, 0
    model.eval()
    model.to(device)
    with torch.no_grad():
        for w1, w2, target, l1, l2 in test_loader:
            target = torch.tensor(target)
            inputs = [
                w1.to(device),
                w2.to(device),
                l1.to(device),
                l2.to(device)]

            h1 = torch.zeros(
                (model.encoder_layers,
                 w1.size(0),
                    model.hidden_size)).to(device)
            h2 = torch.zeros(
                (model.encoder_layers,
                 w1.size(0),
                    model.hidden_size)).to(device)
            out, _, _ = model(inputs, h1, h2)
            preds = torch.squeeze(out)
            total += len(target)
            total_mse += metrics.mean_squared_error(target, preds.cpu())
            total_mae += metrics.mean_absolute_error(target, preds.cpu())
            total_r2 += metrics.r2_score(target, preds.cpu())
    total_mse /= n_batches
    total_mae /= n_batches
    total_r2 /= n_batches
    print(
        f"Evaluated {total} - MSE: {total_mse:.4f}\tMAE: {total_mae:.4f}\tR2: {total_r2:.4f}")
