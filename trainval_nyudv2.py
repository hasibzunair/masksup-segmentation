import os, sys
import random
import time

import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import jaccard_score
from torch.utils.data import DataLoader

from helpers import Logger
from dataset import NYUDV2_dataloader
from models.unetplusplus import NestedUNet


"""Train and evaluate masksup on NYUDv2 dataset"""

########## Reproducibility ##########

SEED = 0
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Load color map
cmap = np.load("datasets/NYUDV2/cmap.npy")

########## Get Args ##########

parser = argparse.ArgumentParser(description="settings")
# configuration
parser.add_argument("--exp_name", default="baseline")
# dataset
parser.add_argument("--dataset", default="isic2018", type=str)
# model
parser.add_argument("--alpha", default=1, type=float)
parser.add_argument("--beta", default=1, type=float)
parser.add_argument("--gamma", default=1, type=float)
args = parser.parse_args()

print(args)

########## Setup ##########

# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# Log folder
EXPERIMENT_NAME = "nyu_reproduce"

ROOT_DIR = os.path.abspath(".")
LOG_PATH = os.path.join(ROOT_DIR, "logs", EXPERIMENT_NAME)

if not os.path.exists(os.path.join(ROOT_DIR, "logs")):
    os.mkdir(os.path.join(ROOT_DIR, "logs"))

if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)
    os.mkdir(os.path.join(LOG_PATH, "samples"))
    os.mkdir(os.path.join(LOG_PATH, "samples", "masked_imgs_cb"))
    os.mkdir(os.path.join(LOG_PATH, "samples", "masked_preds_cb"))
    os.mkdir(os.path.join(LOG_PATH, "samples", "masked_imgs_cb_ts"))
    os.mkdir(os.path.join(LOG_PATH, "samples", "masked_preds_cb_ts"))

# save config in log file
sys.stdout = Logger(os.path.join(LOG_PATH, "log_train.txt"))

########## Load data ##########
train_dataset = NYUDV2_dataloader("datasets/NYUDV2")
test_dataset = NYUDV2_dataloader("datasets/NYUDV2", is_train=False)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

print("Training on {} batches/samples".format(len(train_dataloader)))
print("Testing on {} batches/samples".format(len(test_dataloader)))

dt = next(iter(train_dataloader))
x = dt["image"]
y = dt["mask"]

print("Sample: ", x[0][:, :10][0][0][:3])

########## Get model ##########

# Define model
model = None
model = NestedUNet(num_classes=40)

# Send to GPU
model = model.to(DEVICE)
print(model)

# All parameters
all_params = sum(p.numel() for p in model.parameters())
print("All parameters ", all_params)

# Trainable parameters
all_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable parameters ", all_train_params)

########## Setup optimizer and loss ##########
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=int(0.3 * 200), gamma=0.1, verbose=True
)

criterion_mse = nn.MSELoss()
criterion = nn.CrossEntropyLoss()

########## Trainer and validation functions ##########


def train(model, epoch):
    """
    Trains a segmentation model.
    """
    print("Trains a segmentation model.")

    train_loss = 0

    model.train()
    for batch_idx, data in enumerate(train_dataloader):
        data1, data2, target = (
            data["image"].to(DEVICE),
            data["partial_image1"].to(DEVICE),
            data["mask"].to(DEVICE),
        )

        # Make prediction
        output1 = model.forward(data1.float())

        # Compute loss
        loss = criterion(output1, target.long())
        train_loss += loss.item()

        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_dataloader)
    train_losses.append(train_loss)
    print("Average Training Loss: {:.3f}".format(train_loss))


def train_context_branch(model, epoch, save_masks=True):
    """
    Trains a segmentation model using context branch in a siamese style.
    """

    print("Trains a segmentation model using context branch in a siamese style.")

    model.train()

    train_loss = 0

    for batch_idx, data in enumerate(train_dataloader):
        data1, data2, target = (
            data["image"].to(DEVICE),
            data["partial_image1"].to(DEVICE),
            data["mask"].to(DEVICE),
        )

        # Save masked image
        if save_masks:
            masked_img = data2[0]
            masked_img = (masked_img.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
            masked_img = (masked_img * 255).astype(np.uint8)
            masked_img = cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR)
            # masked_img[masked_img==127] = 0
            cv2.imwrite(
                "{}/samples/masked_imgs_cb/ep{}_b{}.png".format(
                    LOG_PATH, epoch, batch_idx
                ),
                masked_img,
            )

        # Make predictions
        output1 = model.forward(data1.float())
        output2 = model.forward(data2.float())

        # Compute loss based on two outputs
        loss1 = criterion(output1.float(), target.long())
        loss2 = criterion(output2.float(), target.long())

        # Loss coefficients
        alpha = 1
        beta = 1

        # Total loss
        loss = alpha * loss1 + beta * loss2
        train_loss += loss.item()

        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_dataloader)
    train_losses.append(train_loss)
    print("Average Training Loss: {:.3f}".format(train_loss))
    print("With CB: ", alpha, beta)


def train_context_branch_with_task_sim(model, epoch, save_masks=True):
    """
    Trains a segmentation model using context branch (CB) and task similarity (TS) constraint.
    """

    print(
        "Trains a segmentation model using context branch (CB) and task similarity (TS) constraint."
    )

    model.train()

    train_loss = 0

    for batch_idx, data in enumerate(train_dataloader):
        data1, data2, target = (
            data["image"].to(DEVICE),
            data["partial_image1"].to(DEVICE),
            data["mask"].to(DEVICE),
        )

        # Save masked image
        if save_masks:
            masked_img = data2[0]
            masked_img = (masked_img.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
            masked_img = (masked_img * 255).astype(np.uint8)
            masked_img = cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR)
            # masked_img[masked_img==127] = 0
            cv2.imwrite(
                "{}/samples/masked_imgs_cb_ts/ep{}_b{}.png".format(
                    LOG_PATH, epoch, batch_idx
                ),
                masked_img,
            )

        # Make predictions
        output1 = model.forward(data1.float())
        output2 = model.forward(data2.float())

        # Compute loss based on two outputs, and maximize similarity
        loss1 = criterion(output1.float(), target.long())
        loss2 = criterion(output2.float(), target.long())

        output1 = torch.softmax(output1, dim=1).argmax(dim=1)
        output2 = torch.softmax(output2, dim=1).argmax(dim=1)
        loss3 = criterion_mse(output1.float(), output2.float())

        # Loss coefficients
        alpha = 1
        beta = 1
        gamma = 1

        # Total loss
        loss = alpha * loss1 + beta * loss2 + gamma * loss3
        train_loss += loss.item()

        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_dataloader)
    train_losses.append(train_loss)
    print("Average Training Loss: {:.3f}".format(train_loss))
    print("With CB + TS: ", alpha, beta, gamma)


def test(model):
    model.eval()

    with torch.no_grad():
        test_loss = 0
        jaccard = 0

        for data in test_dataloader:
            data, target = data["image"].to(DEVICE), data["mask"].to(DEVICE)
            output = model(data.float())
            test_loss += criterion(output.float(), target.long()).item()
            output = (
                torch.softmax(output, dim=1)
                .argmax(dim=1)[0]
                .float()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            jc = jaccard_score(
                target.squeeze().data.cpu().numpy().flatten(),
                output.flatten(),
                average="micro",
            )
            jaccard += jc

        test_loss /= len(test_dataloader)
        jaccard /= len(test_dataloader)

        losses.append(test_loss)
        jacs.append(jaccard)

        print("Average Loss: {:.3f}".format(test_loss))
        print("Jaccard Index / IoU : {:.3f}".format(jaccard * 100))
        print("==========================================")
        print("==========================================")
        return jaccard


########## Train and validate ##########
train_losses = []
losses = []
jacs = []
dices = []
score = 0
best_score = 0

start_time = time.time()
N_EPOCHS = 200
for epoch in range(1, N_EPOCHS):
    # Train and eval
    print("Epoch: {}".format(epoch))

    ########## Trainer type ##########
    # 1. Base
    # train(model, epoch)
    # 2. Base with context branch
    # train_context_branch(model, epoch)
    # 3. MaskSup
    train_context_branch_with_task_sim(model, epoch)

    score = test(model)
    scheduler.step()

    if score > best_score:
        # Save predictions
        if not os.path.exists(os.path.join(LOG_PATH, "vis")):
            os.mkdir(os.path.join(LOG_PATH, "vis"))
            os.mkdir(os.path.join(LOG_PATH, "vis", "imgs"))
            os.mkdir(os.path.join(LOG_PATH, "vis", "gts"))
            os.mkdir(os.path.join(LOG_PATH, "vis", "preds"))

        for batch_idx, data in enumerate(test_dataloader):
            img, target = data["image"].to(DEVICE), data["mask"].to(DEVICE)
            output = model(img.float())

            img = (img[0].permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
            img = (img * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            gt = target.squeeze().data.cpu().numpy()
            gt = cmap[gt]
            output = (
                torch.softmax(output, dim=1)
                .argmax(dim=1)[0]
                .float()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            pred = cmap[output]

            cv2.imwrite(
                os.path.join(LOG_PATH, "vis", "imgs/") + str(batch_idx) + ".png", img
            )
            cv2.imwrite(
                os.path.join(LOG_PATH, "vis", "gts/") + str(batch_idx) + ".png", gt
            )
            cv2.imwrite(
                os.path.join(LOG_PATH, "vis", "preds/") + str(batch_idx) + ".png", pred
            )

        # Save model
        print("########Saving model at IoU/Jaccard={:.3f}########".format(score))
        torch.save(model.state_dict(), "{}/{}.pth".format(LOG_PATH, EXPERIMENT_NAME))
        best_score = score

end_time = time.time()
print("--- Time taken to train : %s hours ---" % ((end_time - start_time) // 3600))
print("--- Time taken to train : %s mins ---" % ((end_time - start_time) // 60))

print("Max Jaccard/IoU ", max(jacs))

########## Save logs ##########

# Save losses
losses = np.array(losses)
np.savetxt("{}/{}_loss.txt".format(LOG_PATH, EXPERIMENT_NAME), losses, delimiter=",")
train_losses = np.array(train_losses)
np.savetxt(
    "{}/{}_train_loss.txt".format(LOG_PATH, EXPERIMENT_NAME),
    train_losses,
    delimiter=",",
)
jacs = np.array(jacs)
np.savetxt("{}/{}_jacs.txt".format(LOG_PATH, EXPERIMENT_NAME), jacs, delimiter=",")

report = {}

report["Max Jaccard = "] = "{:.5f}".format(max(jacs))

with open("{}/{}_bests.txt".format(LOG_PATH, EXPERIMENT_NAME), "w") as f:
    for k, v in report.items():
        f.write(str(k))
        # f.write("--->")
        f.write(str(v))
        # new line
        f.write("\n")
f.close()

########## Plot curves ##########

# b, g, r, y, o, -g, -m,
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.plot(train_losses, linewidth=4)
plt.plot(losses, linewidth=4)
plt.title("{} loss".format("Exp name"))
# plt.ylabel('Loss')
plt.xlabel("Epoch")
plt.legend(["train_loss", "loss"], loc="upper left")
plt.grid(True)
# Plot training & validation iou_score values
plt.subplot(122)
plt.plot(jacs, linewidth=4)
# plt.plot(dices,linewidth=4)
# plt.title('{} IOU score'.format(experiment_name))
# plt.ylabel('iou_score')
plt.xlabel("Epoch")
plt.grid(True)
plt.legend(["Jaccard"], loc="upper left")
plt.savefig("{}/{}_graph.png".format(LOG_PATH, EXPERIMENT_NAME), dpi=300)
# plt.show()
