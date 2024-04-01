import argparse
import torch
import random
import numpy as np
import os
# from visdom import Visdom
import math

import torchsummary
# import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from utils import log, data_generator, show
from utils.adjust_lr import adjust_learning_rate
import model.builder as builder


# --------------Get parameters list-------------------
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='./data/processed_data', type=str, help='path of dataset')
parser.add_argument('--num_workers', default=0, type=int, help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, help='the number of training epochs')
parser.add_argument('--batch_size', default=128, type=int, help='total batchsize of all GPUs or single GPU')
parser.add_argument('--learning_rate', default=5e-4, type=float, help='the initial learning rate of optimizer')
parser.add_argument('--display_step', default=1, type=int, help="How many steps to print once")
parser.add_argument('--device', default='cuda', type=str, help='the training device, cpu or cuda')
parser.add_argument('--log_save_dir', default='./log', type=str, help='the path of log files')
parser.add_argument('--seed', default=42, type=int, help='seed value')
parser.add_argument('--head', default=8, type=int, help="the number of head in multihead-attention module")
parser.add_argument('--dropout', default=0.35, type=float, help="the probability of dropout")
parser.add_argument('--d_ff', default=1024, type=int, help="the dim of the hidden layer in feedforward")
parser.add_argument('--temporal_dim', default=512, type=int, help="the dim of sequence")
parser.add_argument('--feature_dim', default=16, type=int, help="the dim of feature")
parser.add_argument('--proj_dim', default=512, type=int, help="the hidden dim of the projector")
parser.add_argument('--pred_dim', default=512, type=int, help="the hidden dim of the predictor")
parser.add_argument('--pretext_dim', default=2048, type=int, help="the feature dimension of extracted in pretext task")
parser.add_argument('--channel_dim', default=32, type=int, help="the dim of channel")
parser.add_argument('--ATE_num', default=2, type=int, help="the number of the ATE_layer")
# get parameters
args = parser.parse_args()


# ------------------Define single_main() function-------------------

# use single device
def main():

    # <parse out parameters>
    data_path = args.data_path

    # <set logger, and print the message>
    logger_ob = log.Logger(log_save_dir=args.log_save_dir)
    logger = logger_ob.set_logger()
    logger_ob.display(logger)

    # <set seed>
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        np.random.seed(args.seed)
        logger.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.device is not None:
        logger.debug("=> You have chosen to use {} to train!".format(args.device))

    # <load the dataset>
    # suppose the data of .pt file is a Ordered dictionary
    train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    valid_dataset = torch.load(os.path.join(data_path, "valid.pt"))
    # test_dataset = torch.load(os.path.join(data_path, "test.pt"))
    train_loader = data_generator.data_generator(train_dataset, args)
    valid_loader = data_generator.data_generator(valid_dataset, args)
    # test_loader = data_generator.data_generator(test_dataset, args)
    logger.debug("=> Data has successfully loaded!")

    # <create model>
    logger.debug("=> Creating model...")
    model = builder.ATE(args).to(args.device)
    if args.device == "cuda":
        torchsummary.summary(model, [(512, 16), [512, 16]])
    else:
        print(model)

    # <define criterion and optimizer>
    logger.debug("=> Creating criterion and optimizer...")
    criterion = nn.CosineSimilarity(dim=1).to(args.device)
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=3e-4)

    # <train and valid for each epoch>
    logger.debug("################## Training is Beginning! #########################")
    logger.debug("=> Start train for {} epochs...".format(args.epochs))
    # viz = Visdom(port=8097)
    train_loss_list = []
    valid_loss_list = []
    init_lr = args.learning_rate
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args)
        train_loss = train(model, train_loader, criterion, optimizer, args)
        valid_loss = valid(model, valid_loader, criterion, optimizer, args)
        # opts = {
        #     "title": 'Pretext_loss',
        #     "xlabel": 'Epochs',
        #     "ylabel": 'Loss',
        #     "width": 1600,
        #     "height": 1600,
        #     "legend": ["train_loss", "valid_loss"]
        # }
        # viz.line(X=[epoch], Y=[[train_loss, valid_loss]],win='Pretext_loss', update='append', opts=opts)
        train_loss_list.append(train_loss.item())
        valid_loss_list.append(valid_loss.item())
        if epoch % args.display_step == 0:
            logger.debug(f'Epoch : {epoch}\n'
                         f'Train Loss     : {train_loss:.4f}\t | \tValid Loss     : {valid_loss:.4f}')
    # record the loss in the process of train and valid
    train_loss_array = np.array(train_loss_list)
    valid_loss_array = np.array(valid_loss_list)

    # <save the losses>
    os.makedirs("./saved/pre/saved_losses",exist_ok=True)
    ckp_loss = {"train_loss_array": train_loss_array, "valid_loss_array": valid_loss_array}
    losses_save_path = "./saved/pre/saved_losses/ATEnum{}_dropout{}_lr{}.pt".format(args.ATE_num, args.dropout, args.learning_rate)
    torch.save(ckp_loss, losses_save_path)
    logger.debug("=> Losses has been saved at {}!".format(losses_save_path))

    # <plot train & valid image>
    show.plot_image(train_loss_array, valid_loss_array, args)

    # <save trained model's parameters>
    os.makedirs("./saved/pre/saved_models",exist_ok=True)
    models_save_path = "./saved/pre/saved_models/ATEnum{}_dropout{}_lr{}.pt".format(args.ATE_num, args.dropout, args.learning_rate)
    torch.save(model.state_dict(), models_save_path)
    logger.debug("=> Model has been saved at {}!".format(models_save_path))

    logger.debug(
        "################## Training is Done! #########################")


def valid(model, valid_loader, criterion, optimizer, args):
    valid_loss = []
    model.eval()
    with torch.no_grad():
        for batch_index, (aug1, aug2) in enumerate(valid_loader):
            p1, p2, z1, z2 = model(aug1, aug2)
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean())*0.5
            valid_loss.append(loss.item())
        valid_loss = torch.tensor(valid_loss).mean()
    return valid_loss


def train(model, train_loader, criterion, optimizer, args):
    train_loss = []
    model.train()

    for batch_index, (aug1, aug2) in enumerate(train_loader):
        p1, p2, z1, z2 = model(aug1, aug2)
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean())*0.5
        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = torch.tensor(train_loss).mean()
    return train_loss

if __name__ == "__main__":
    main()
