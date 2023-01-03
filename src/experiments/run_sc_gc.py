import torch
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt

avail_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, loader, optimizer, loss_fun):
    model.train()
    loss_all = 0

    for data in loader:
        data = data.to(avail_device)
        optimizer.zero_grad()
        loss = loss_fun(model(data), data.y).to(avail_device)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()

    return loss_all / len(loader.dataset)


def val(model, loader, loss_fun, y_idx=0):
    model.eval()
    loss_all = 0

    for data in loader:
        data = data.to(avail_device)
        loss_all += loss_fun(model(data), data.y).item()

    return loss_all / len(loader.dataset)


def test(model, loader):
    model.eval()
    total_err = 0

    for data in loader:
        data = data.to(avail_device)
        pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()

    return correct / len(loader.dataset)

def run_sc_model_gc(
    model,
    dataset_tr,
    dataset_val,
    batch_size=32,
    lr=0.0001,
    epochs=300,
    nb_reruns=5,
):

    plot_train_loss = []
    plot_val_loss = []
    plot_epoch = []

    loss_fun = torch.nn.BCEWithLogitsLoss()

    print("----------------- Predicting bug presence -----------------")
    all_val_loss = np.zeros(nb_reruns,)

    for rerun in range(nb_reruns): 
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # Made static

        val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False) # no shufflng for training
        train_loader = DataLoader( # Shuffle for training
            dataset_tr, batch_size=batch_size, shuffle=True
        )  

        print(
            "---------------- "
            + ": Re-run {} ----------------".format(rerun)
        )

        best_val_loss = 100000
 
        for epoch in range(1, epochs + 1):
            # lr = scheduler.optimizer.param_groups[0]['lr']  # Same as GC
            train_loss = train(
                model, train_loader, optimizer, loss_fun
            )
            val_loss = val(model, val_loader, loss_fun)
            # scheduler.step(val_mse_sum)
            if best_val_loss >= val_loss:  # Improvement in validation loss
                best_val_loss = val_loss


            # ======================================
            # Plotting
            # ======================================
            
            plot_train_loss.append(train_loss)
            plot_epoch.append(epoch)
            plot_val_loss.append(val_loss)


            print(
                "Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, "
                "Val Loss: {:.7f}".format(
                    epoch, lr, train_loss, val_loss
                )
            )

        all_val_loss[rerun] = best_val_loss

    # Calculate mean and standard deviation of validation results
    avg_val_loss = all_val_loss.mean()
    std_val_loss = np.std(all_val_loss)


    torch.save(model, "../model_eito.pt")

    plt.plot(plot_epoch, plot_train_loss, label = "training loss")
    plt.plot(plot_epoch, plot_val_loss, label = "validation loss")
    plt.legend()
    plt.show()

    print("---------------- Final Result ----------------")
    print("Validation -- Mean: " + str(avg_val_loss) + ", Std: " + str(std_val_loss))

