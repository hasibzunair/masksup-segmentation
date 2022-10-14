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

from torch.utils.data import DataLoader

from helpers import Logger
from dataset import GLAS_dataloader, POLYPS_dataloader
from metrics import calculate_metric_percase
from models.LeViTUNet384 import Build_LeViT_UNet_384

"""Train and evaluate masksup on GLaS or Kvasir & CVC-ClinicDB datasets"""

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
# EXPERIMENT_NAME = args.exp_name+"_"+"a"+str(args.alpha)+"b"+str(args.beta)+"g"+str(args.gamma)+"_"+args.dataset #"levit192_isic2018"
EXPERIMENT_NAME = "glas_reproduce"

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

if "glas" in EXPERIMENT_NAME:
    print("Loading GLAS dataset")
    train_dataset = GLAS_dataloader("datasets/GLAS")
    test_dataset = GLAS_dataloader("datasets/GLAS", is_train=False)
else:
    print("Loading POLYP dataset")
    train_dataset = POLYPS_dataloader("datasets/POLYPS")
    test_dataset = POLYPS_dataloader("datasets/POLYPS", is_train=False)


train_dataloader = DataLoader(
    train_dataset, batch_size=8, shuffle=True, num_workers=8
)  # 8
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

print("Training on {} batches/samples".format(len(train_dataloader)))
print("Testing on {} batches/samples".format(len(test_dataloader)))

dt = next(iter(train_dataloader))
x = dt["image"]
y = dt["mask"]

print("Sample: ", x[0][:, :10][0][0][:3])

########## Get model ##########

# Define model
model = Build_LeViT_UNet_384(num_classes=1, pretrained=True)

# Send to GPU
model = model.to(DEVICE)
print(model)

# All parameters
all_params = sum(p.numel() for p in model.parameters())
print("All parameters ", all_params)

# Trainable parameters
all_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable parameters ", all_train_params)

########## Setup optimizer and loss and loss scaling for AMP ##########

optimizer = optim.Adam(
    model.parameters(), lr=1e-4, weight_decay=1e-5, amsgrad=True
)  # prev 1e-4
amp = True
grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

criterion = (
    nn.BCEWithLogitsLoss()
)  # loss combines a Sigmoid layer and the BCELoss in one single class
criterion_mse = nn.MSELoss()

########## Trainer and validation functions ##########


def train(model, epoch):
    """
    Trains a segmentation model.
    """
    print("Trains a segmentation model.")

    assert model is not None, f"Should be a PyTorch model, got: {model}"

    model.train()
    for batch_idx, data in enumerate(train_dataloader):
        data1, data2, target = (
            data["image"].to(DEVICE),
            data["partial_image1"].to(DEVICE),
            data["mask"].to(DEVICE),
        )
        output1 = model.forward(data1.float())

        # Compute loss based on two outputs
        loss = criterion(output1.float(), target.float())

        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_context_branch(model, epoch, save_masks=True):
    """
    Trains a segmentation model using context branch in a siamese style.
    """

    print("Trains a segmentation model using context branch in a siamese style.")

    assert model is not None, f"Should be a PyTorch model, got: {model}"

    model.train()
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
        loss1 = criterion(output1.float(), target.float())
        loss2 = criterion(output2.float(), target.float())

        # Loss coefficients
        alpha = 1
        beta = 1

        # Total loss
        loss = alpha * loss1 + beta * loss2

        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("With CB: ", alpha, beta)


def train_context_branch_with_task_sim(model, epoch, save_masks=True):
    """
    Trains a segmentation model using context branch (CB) and task similarity (TS) constraint.
    """

    print(
        "Trains a segmentation model using context branch (CB) and task similarity (TS) constraint."
    )

    model.train()
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

        with torch.cuda.amp.autocast(enabled=amp):
            # Make predictions
            output1 = model.forward(data1.float())
            output2 = model.forward(data2.float())

            # Compute loss based on two outputs, and maximize similarity
            loss1 = criterion(output1.float(), target.float())
            loss2 = criterion(output2.float(), target.float())
            loss3 = criterion_mse(
                torch.sigmoid(output1.float()), torch.sigmoid(output2.float())
            )

        # Loss coefficients
        alpha = 1
        beta = 1
        gamma = 1

        # Total loss
        loss = alpha * loss1 + beta * loss2 + gamma * loss3

        # Update
        optimizer.zero_grad(set_to_none=True)
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()

    print("With CB + TS: ", alpha, beta, gamma)


def test(model):
    model.eval()

    with torch.no_grad():
        test_loss = 0
        jaccard = 0
        dice = 0

        for data in test_dataloader:
            data, target = data["image"].to(DEVICE), data["mask"].to(DEVICE)
            output = model(data.float())
            test_loss += criterion(output.float(), target.float()).item()
            dc, jc, _ = calculate_metric_percase(output, target)
            jaccard += jc
            dice += dc
        test_loss /= len(test_dataloader)
        jaccard /= len(test_dataloader)
        dice /= len(test_dataloader)

        losses.append(test_loss)
        jacs.append(jaccard)
        dices.append(dice)

        print("Average Loss: {:.3f}".format(test_loss))
        print("Jaccard Index / IoU : {:.3f}".format(jaccard * 100))
        print("Dice Coefficient / F1 : {:.3f}".format(dice * 100))
        print("==========================================")
        print("==========================================")
        return jaccard


########## Train and validate ##########

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

    if score > best_score:
        # Save predictions
        if not os.path.exists(os.path.join(LOG_PATH, "vis")):
            os.mkdir(os.path.join(LOG_PATH, "vis"))
            os.mkdir(os.path.join(LOG_PATH, "vis", "imgs"))
            os.mkdir(os.path.join(LOG_PATH, "vis", "gts"))
            os.mkdir(os.path.join(LOG_PATH, "vis", "preds"))

        for batch_idx, data in enumerate(test_dataloader):
            img, target = data["image"].to(DEVICE), data["mask"].to(DEVICE)
            output = torch.sigmoid(model(img.float()))

            img = (img[0].permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
            img = (img * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            gt = target.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy()
            gt = (gt * 255).astype(np.uint8)
            gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)

            pred = output.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy() > 0.5
            pred = (pred * 255).astype(np.uint8)
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

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

print("Max Jaccard/IoU and Dice/F1 scores: ", max(jacs), " and ", max(dices))

########## Save logs ##########

# Save losses
losses = np.array(losses)
np.savetxt("{}/{}_loss.txt".format(LOG_PATH, EXPERIMENT_NAME), losses, delimiter=",")
jacs = np.array(jacs)
np.savetxt("{}/{}_jacs.txt".format(LOG_PATH, EXPERIMENT_NAME), jacs, delimiter=",")
dices = np.array(dices)
np.savetxt("{}/{}_dices.txt".format(LOG_PATH, EXPERIMENT_NAME), dices, delimiter=",")

report = {}

report["Max Jaccard = "] = "{:.5f}".format(max(jacs))
report["Max Dice = "] = "{:.5f}".format(max(dices))

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
plt.plot(losses, linewidth=4)
plt.title("{} loss".format("Exp name"))
# plt.ylabel('Loss')
plt.xlabel("Epoch")
plt.legend(["loss"], loc="upper left")
plt.grid(True)
# Plot training & validation iou_score values
plt.subplot(122)
plt.plot(jacs, linewidth=4)
plt.plot(dices, linewidth=4)
# plt.title('{} IOU score'.format(experiment_name))
# plt.ylabel('iou_score')
plt.xlabel("Epoch")
plt.grid(True)
plt.legend(["Jaccard", "Dice"], loc="upper left")
plt.savefig("{}/{}_graph.png".format(LOG_PATH, EXPERIMENT_NAME), dpi=300)
# plt.show()
