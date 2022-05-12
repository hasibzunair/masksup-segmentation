import os,sys,inspect
import random
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from torch.autograd import Variable

from helpers import Logger
from dataset import ISIC2018_dataloader, CVCLINICDB_dataloader
from models.unet import build_unet
from models.LeViTUNet128s import Build_LeViT_UNet_128s
from models.LeViTUNet192 import Build_LeViT_UNet_192
from models.LeViTUNet384 import Build_LeViT_UNet_384
from models.bigcn import ODOC_seg_edge
from models.unetplusplus import NestedUNet

"""Training script"""

########## Reproducibility ##########

random.seed(0)
os.environ['PYTHONHASHSEED'] = str(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


########## Get Args ##########

# def Args():
#     parser = argparse.ArgumentParser(description="settings")
#     # configuration
#     parser.add_argument("--exp_name", default="baseline")
#     # dataset
#     parser.add_argument("--dataset", default="isic2018", type=str)
#     # model
#     parser.add_argument("--alpha",default=1, type=float)
#     parser.add_argument("--beta",default=1, type=float)
#     parser.add_argument("--gamma",default=1, type=float)
    
# #     parser.add_argument("--cutmix", default=None, type=str) # the path to load cutmix-pretrained backbone
# #     # dataset
# #     parser.add_argument("--dataset", default="voc07", type=str)
# #     parser.add_argument("--num_cls", default=20, type=int)
# #     parser.add_argument("--train_aug", default=["randomflip", "resizedcrop"], type=list)
# #     parser.add_argument("--test_aug", default=[], type=list)
# #     parser.add_argument("--img_size", default=448, type=int)
# #     parser.add_argument("--batch_size", default=16, type=int)
# #     # optimizer, default SGD
# #     parser.add_argument("--lr", default=0.01, type=float)
# #     parser.add_argument("--momentum", default=0.9, type=float)
# #     parser.add_argument("--w_d", default=0.0001, type=float, help="weight_decay")
# #     parser.add_argument("--warmup_epoch", default=2, type=int)
# #     parser.add_argument("--total_epoch", default=30, type=int)
# #     parser.add_argument("--print_freq", default=100, type=int)
#     args = parser.parse_args()
#     return args

parser = argparse.ArgumentParser(description="settings")
# configuration
parser.add_argument("--exp_name", default="baseline")
# dataset
parser.add_argument("--dataset", default="isic2018", type=str)
# model
parser.add_argument("--alpha",default=1, type=float)
parser.add_argument("--beta",default=1, type=float)
parser.add_argument("--gamma",default=1, type=float)
args = parser.parse_args()

print(args)

########## Setup ##########

# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# Log folder
#EXPERIMENT_NAME = args.exp_name+"_"+"a"+str(args.alpha)+"b"+str(args.beta)+"g"+str(args.gamma)+"_"+args.dataset #"levit192_isic2018"
EXPERIMENT_NAME = "unet_cvcdb"

ROOT_DIR = os.path.abspath(".")
LOG_PATH = os.path.join(ROOT_DIR, "logs", EXPERIMENT_NAME)

if not os.path.exists(os.path.join(ROOT_DIR, "logs")):
    os.mkdir(os.path.join(ROOT_DIR, "logs"))
    
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)
    
# save config in log file
sys.stdout = Logger(os.path.join(LOG_PATH, 'log_train.txt'))


########## Load data ##########

train_dataset = CVCLINICDB_dataloader("datasets/CVCLINICDB") # ISIC2018
test_dataset = CVCLINICDB_dataloader("datasets/CVCLINICDB", is_train=False) # ISIC2018

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

dt = next(iter(train_dataloader))
x = dt["image"]
y = dt["mask"]

print("Sample: ", x[0][:,:10][0][0][:3])


########## Get model ##########

# Define model
model = build_unet()
model = NestedUNet()
#model = Build_LeViT_UNet_128s(num_classes=1, pretrained=True)
#model = Build_LeViT_UNet_192(num_classes=1, pretrained=True)
#model = Build_LeViT_UNet_384(num_classes=1, pretrained=True)
#model = ODOC_seg_edge()

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

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.BCEWithLogitsLoss() # loss combines a Sigmoid layer and the BCELoss in one single class
criterion_mse = nn.MSELoss()

########## Trainer and validation functions ##########

def train(model, epoch):
    """
    Trains a segmentation model.
    """
    print("Trains a segmentation model.")
    
    model.train()
    for batch_idx, data in enumerate(train_dataloader):
        data1, data2, target = data["image"].to(DEVICE), data["partial_image1"].to(DEVICE), data["mask"].to(DEVICE)
        output1 = model.forward(data1.float())
       
        # Compute loss based on two outputs
        loss = criterion(output1.float(), target.float())
        
        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_context_branch(model, epoch):
    """
    Trains a segmentation model using context branch in a siamese style. 
    """
    
    print("Trains a segmentation model using context branch in a siamese style.")
    
    model.train()
    for batch_idx, data in enumerate(train_dataloader):
        data1, data2, target = data["image"].to(DEVICE), data["partial_image1"].to(DEVICE), data["mask"].to(DEVICE)
        output1 = model.forward(data1.float())
        output2 = model.forward(data2.float())
        
        # Compute loss based on two outputs
        loss1 = criterion(output1.float(), target.float())
        loss2 = criterion(output2.float(), target.float())
        # Loss coefficients
        alpha = 1 #args.alpha
        beta = 1 #args.beta
        # Total loss
        loss = alpha * loss1 + beta * loss2
        
        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print("With CB: ", alpha, beta)


def train_context_branch_with_task_sim(model, epoch):
    """
    Trains a segmentation model using context branch (CB) and task similarity (TS) constraint. 
    """
    
    print("Trains a segmentation model using context branch (CB) and task similarity (TS) constraint.")
    
    model.train()
    for batch_idx, data in enumerate(train_dataloader):
        data1, data2, target = data["image"].to(DEVICE), data["partial_image1"].to(DEVICE), data["mask"].to(DEVICE)
        output1 = model.forward(data1.float())
        output2 = model.forward(data2.float())

        # Compute loss based on two outputs, and enforce similarity
        loss1 = criterion(output1.float(), target.float())
        loss2 = criterion(output2.float(), target.float())
        loss3 = criterion_mse(torch.sigmoid(output1.float()), torch.sigmoid(output2.float()))
        
        # Loss coefficients
        # 0.4, 0.2, 0.4 with LeViT128 on ISIC is best
        alpha = 25 #0.4
        beta = 1 #0.2
        gamma = 50 #0.4
        
        #  
        # 
        
        # Notes (Unet isic2018)
        # 1,1,1 -> ( BEST)Max jaccard and dice:  0.8136580522714527  and  0.8864272972888932 (unet_cb_ts_isic2018)
        # 0.7,0.2,0.3 -> Max jaccard and dice:  0.8094201231544016  and  0.8833294555615818
        # 0.5,0.2,0.3 -> Max jaccard and dice:  0.8131758432820985  and  0.8857327138584684
        # 0.4,0.2,0.4 -> ???
        # 0.7,0.2,0.3 ->
        
        
        # Total loss
        loss = alpha * loss1 + beta * loss2 + gamma * loss3
        
        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
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
            
            output = torch.sigmoid(output) # Turn activations into probabilities by feeding through sigmoid
            gt = target.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy()
            pred = output.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy() > 0.5

            intersection = pred * gt
            union = pred + gt - intersection
            jaccard += (np.sum(intersection)/np.sum(union))  
            dice += (2. * np.sum(intersection) ) / (np.sum(pred) + np.sum(gt))
    
        test_loss /= len(test_dataloader)
        jaccard /= len(test_dataloader)
        dice /= len(test_dataloader)

        losses.append(test_loss)
        jacs.append(jaccard)
        dices.append(dice)


        print('Average Loss: {:.3f}'.format(test_loss))
        print('Jaccard Index : {:.3f}'.format(jaccard * 100))
        print('Dice Coefficient : {:.3f}'.format(dice * 100))
        print('==========================================')
        print('==========================================')
        return dice

    
########## Train and validate ##########

losses = []
jacs = []
dices = []
score = 0
best_score = 0

start_time = time.time()
N_EPOCHS = 100
for epoch in range(1, N_EPOCHS):
    # Train and eval
    print("Epoch: {}".format(epoch))
    
    # Trainer type
    train(model, epoch)
    #train_context_branch(model, epoch)
    #train_context_branch_with_task_sim(model, epoch)
    score = test(model)
    
    # Save best model
    if score > best_score:
        print("Saving model at dice={:.3f}".format(score))
        torch.save(model.state_dict(), '{}/{}.pth'.format(LOG_PATH, EXPERIMENT_NAME))
        best_score = score

end_time = time.time()
print("--- Time taken to train : %s hours ---" % ((end_time - start_time)//3600))
print("--- Time taken to train : %s mins ---" % ((end_time - start_time)//60))

print("Max jaccard and dice: ", max(jacs)," and ", max(dices))

########## Save logs ##########

# Save losses
losses = np.array(losses)
np.savetxt("{}/{}_loss.txt".format(LOG_PATH, EXPERIMENT_NAME), losses, delimiter=",")
jacs = np.array(jacs)
np.savetxt("{}/{}_jacs.txt".format(LOG_PATH, EXPERIMENT_NAME), jacs, delimiter=",")
dices = np.array(dices)
np.savetxt("{}/{}_dices.txt".format(LOG_PATH, EXPERIMENT_NAME), dices, delimiter=",")

report = {}

report['Max Jaccard = '] = "{:.5f}".format(max(jacs))
report['Max Dice = '] = "{:.5f}".format(max(dices))

with open("{}/{}_bests.txt".format(LOG_PATH, EXPERIMENT_NAME), 'w') as f:
    for k,v in report.items():
            f.write(str(k))
            #f.write("--->")
            f.write(str(v))
            # new line
            f.write("\n")
f.close()

########## Plot curves ##########

# b, g, r, y, o, -g, -m,
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.plot(losses,linewidth=4)
plt.title('{} loss'.format("Exp name"))
#plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['loss'], loc='upper left')
plt.grid(True)
# Plot training & validation iou_score values
plt.subplot(122)
plt.plot(jacs,linewidth=4)
plt.plot(dices,linewidth=4)
#plt.title('{} IOU score'.format(experiment_name))
#plt.ylabel('iou_score')
plt.xlabel('Epoch')
plt.grid(True)
plt.legend(['Jaccard', 'Dice'], loc='upper left')
plt.savefig('{}/{}_graph.png'.format(LOG_PATH, EXPERIMENT_NAME), dpi=300)
#plt.show()
