import argparse
import torch
import random
import numpy as np
import os
from visdom import Visdom
import math

import torchsummary
# import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import model.loader as loader
from utils import log, data_generator_cls, show
from utils.adjust_lr import adjust_learning_rate
import matplotlib.pyplot as plt


# --------------Get parameters list-------------------
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='./data/processed_data', type=str, help='path of dataset')
parser.add_argument('--num_workers', default=0, type=int, help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, help='the number of training epochs')
parser.add_argument('--batch_size', default=32, type=int, help='total batchsize of all GPUs or single GPU')
parser.add_argument('--learning_rate', default=1e-6, type=float, help='the initial learning rate of optimizer')
parser.add_argument('--cls_learning_rate', default=1e-6, type=float, help='the selected lr of pretext task')
parser.add_argument('--pre_slc_learning_rate', default=1e-6, type=float, help='the selected lr of pretext task')
parser.add_argument('--display_step', default=1, type=int, help="How many steps to print once")
parser.add_argument('--multiprocessing', action='store_true', help='whether to choose multiprocessing')
parser.add_argument('--device', default='cuda', type=str, help='the training device, cpu or cuda')
parser.add_argument('--log_save_dir', default='./log', type=str, help='the path of log files')
parser.add_argument('--seed', default=42, type=int, help='seed value')
parser.add_argument('--cls_feature', default='mfccs', type=str, help='the input feature of classification model')
parser.add_argument('--head', default=8, type=int, help="the number of head in multihead-attention module")
parser.add_argument('--dropout', default=0.35, type=float, help="the probability of dropout")
parser.add_argument('--d_ff', default=1024, type=int, help="the dim of the hidden layer in feedforward")
parser.add_argument('--temporal_dim', default=512, type=int, help="the dim of sequence")
parser.add_argument('--feature_dim', default=16, type=int, help="the dim of feature")
parser.add_argument('--proj_dim', default=512, type=int, help="the hidden dim of the projector")
parser.add_argument('--pred_dim', default=512, type=int, help="the hidden dim of the predictor")
parser.add_argument('--pretext_dim', default=2048, type=int, help="the feature dimension of extracted in pretext task")
parser.add_argument('--ATE_num', default=2, type=int, help="the number of the ATE_layer")
parser.add_argument('--channel_dim', default=32, type=int, help="the dim of channel")
parser.add_argument('--mode', default='train_val', type=str, help='the selected mode, train_val or test')
# get parameters
args = parser.parse_args()


def train_val_main():
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
    # {mfccs:[],train_label:[]}
    train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    valid_dataset = torch.load(os.path.join(data_path, "test.pt"))
    train_loader = data_generator_cls.data_generator(train_dataset, args)
    valid_loader = data_generator_cls.data_generator(valid_dataset, args)
    logger.debug("=> Data has successfully loaded!")

    # <create model>
    logger.debug("=> Creating model, freezing ATE.encoder, adding linear_classification_head...")
    model = loader.Lincls(args).to(args.device)
    if args.device == "cuda":
        torchsummary.summary(model, (512, 16))
    else:
        print(model)

    # <define criterion and optimizer>
    logger.debug("=> Creating criterion and optimizer...")
    criterion = nn.CrossEntropyLoss().to(args.device)
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.Adam(parameters, lr=args.learning_rate, weight_decay=3e-4)

    # <train and valid for each epoch>
    logger.debug("################## Training is Beginning! #########################")
    logger.debug("=> Start train for {} epochs...".format(args.epochs))
    # viz = Visdom(port=8097)
    train_loss_list = []
    valid_loss_list = []
    train_acc_list = []
    valid_acc_list = []
    init_lr = args.learning_rate

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args)
        model.cls_head.train()
        train_loss, train_acc = train_cls(model, train_loader, criterion, optimizer, args)
        model.cls_head.eval()
        valid_loss, valid_acc = valid_cls(model, valid_loader, criterion, optimizer, args)
        # opts = {
        #     "title": 'Pretext_loss',
        #     "xlabel": 'Epochs',
        #     "ylabel": 'Loss',
        #     "width": 1600,
        #     "height": 900,
        #     "legend": ["train_loss", "valid_loss"]
        # }
        # viz.line(X=[epoch], Y=[[train_loss, valid_loss]], win='Pretext_loss', update='append',opts=opts)
        train_loss_list.append(train_loss.item())
        valid_loss_list.append(valid_loss.item())
        train_acc_list.append(train_acc.item())
        valid_acc_list.append(valid_acc.item())
        if epoch % args.display_step == 0:
            logger.debug(f'Epoch : {epoch}\n'
                        f'Train_cls Loss: {train_loss:.4f}\tValid_cls Loss: {valid_loss:.4f}\t | \tTrain_cls Acc: {train_acc:.4f}\tValid_cls Acc: {valid_acc:.4f}')
    # record the loss in the process of train and valid
    train_loss_array = np.array(train_loss_list)
    valid_loss_array = np.array(valid_loss_list)
    train_acc_array = np.array(train_acc_list)
    valid_acc_array = np.array(valid_acc_list)

    # <save the losses>
    os.makedirs("./saved/cls/saved_losses",exist_ok=True)
    ckp_loss_acc = {"train_loss_array": train_loss_array, "valid_loss_array": valid_loss_array, "train_acc_array": train_acc_array, "valid_acc_array": valid_acc_array}
    losses_acc_save_path = "./saved/cls/saved_losses/ATEnum{}_lr{}.pt".format(args.ATE_num, args.learning_rate)
    torch.save(ckp_loss_acc, losses_acc_save_path)
    logger.debug("=> Losses_cls has been saved at {}!".format(losses_acc_save_path))

    # <plot train & valid image>
    show.plot_image_cls(train_loss_array, valid_loss_array, train_acc_array, valid_acc_array, args)

    # <save trained model's parameters>
    os.makedirs("./saved/cls/saved_models",exist_ok=True)
    models_save_path = "./saved/cls/saved_models/ATEnum{}_lr{}.pt".format(args.ATE_num, args.learning_rate)
    torch.save(model.state_dict(), models_save_path)
    logger.debug("=> Model_cls has been saved at {}!".format(models_save_path))

    logger.debug(
        "################## Training is Done! #########################")


def train_cls(model, train_loader, criterion, optimizer, args):
    train_loss = []
    train_acc = []

    for batch_index, (feature, label) in enumerate(train_loader):
        feature = feature.to(args.device)
        label = label.to(args.device)
        conf_matrix = torch.zeros(2, 2)

        pre = model(feature).squeeze(1)
        loss = criterion(pre, label)
        pre = torch.argmax(pre, 1)
        conf_matrix = confusion_matrix(pre, label, conf_matrix)
        acc = conf_matrix.diag().sum()/conf_matrix.sum()
        train_acc.append(acc.item())
        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_acc = torch.tensor(train_acc).mean()
    train_loss = torch.tensor(train_loss).mean()
    return train_loss, train_acc


def valid_cls(model, valid_loader, criterion, optimizer, args):
    valid_loss = []
    valid_acc = []
    with torch.no_grad():
        for batch_index, (feature, label) in enumerate(valid_loader):
            feature = feature.to(args.device)
            label = label.to(args.device)
            conf_matrix = torch.zeros(2, 2)

            pre = model(feature).squeeze(1)
            loss = criterion(pre, label)
            pre = torch.argmax(pre, 1)
            conf_matrix = confusion_matrix(pre, label, conf_matrix)
            acc = conf_matrix.diag().sum()/conf_matrix.sum()
            valid_acc.append(acc.item())
            valid_loss.append(loss.item())

        valid_acc = torch.tensor(valid_acc).mean()
        valid_loss = torch.tensor(valid_loss).mean()

    return valid_loss, valid_acc


def test_main():
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
        # cudnn.deterministic = True
        np.random.seed(args.seed)
        logger.warn(
            "You have chosen to seed testing. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your testing considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.device is not None:
        logger.debug("=> You have chosen to use {} to test!".format(args.device))

    # <load the dataset>
    # suppose the data of .pt file is a Ordered dictionary
    # {mfccs:[],train_label:[]}
    test_dataset = torch.load(os.path.join(data_path, "test.pt"))
    test_loader = data_generator_cls.data_generator(test_dataset, args)
    logger.debug("=> Data has successfully loaded!")

    # <create model>
    logger.debug("=> Creating model, freezing ATE.encoder, adding linear_classification_head...")
    model = loader.Lincls_test(args).to(args.device)
    model.eval()
    # models_save_path = "./saved/cls/saved_models/ATEnum{}_lr{}.pt".format(args.ATE_num, args.cls_learning_rate)
    # state = torch.load(models_save_path)
    # model.load_state_dict(state)
    if args.device == "cuda":
        torchsummary.summary(model, (512, 16))
    else:
        print(model)

    # <define criterion and optimizer>
    logger.debug("=> Creating criterion and optimizer...")
    criterion = nn.CrossEntropyLoss().to(args.device)

    # <test for each epoch>
    logger.debug("################## Test is Beginning! #########################")
    # viz = Visdom(port=8097)

    test(model, test_loader, criterion, args)

    logger.debug("################## Test is Done! #########################")


def test(model, test_loader, criterion, args):
    with torch.no_grad():
        conf_matrix = torch.zeros(2, 2)
        for batch_index, (feature, label) in enumerate(test_loader):
            feature = feature.to(args.device)
            label = label.to(args.device)
            # shape:[batch_size, 2]
            pre = model(feature)
            pre = torch.argmax(pre, 1)
            conf_matrix = confusion_matrix(pre, label, conf_matrix)
        plot_matrix(conf_matrix)


def confusion_matrix(preds, labels, conf_matrix):
    # preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def plot_matrix(conf_matrix):
    kinds_num = 2
    labels = ['Healthy', 'COPD']
    acc = conf_matrix.diag().sum()/conf_matrix.sum()

    plt.imshow(conf_matrix, cmap=plt.cm.Blues)

    thresh = conf_matrix.max() * 2 / 3
    for x in range(kinds_num):
        for y in range(kinds_num):
            info = int(conf_matrix[y, x])
            plt.text(x, y, info,
                     verticalalignment='center',
                     horizontalalignment='center',
                     color="white" if info > thresh else "black")
    plt.tight_layout()
    plt.yticks(range(kinds_num), labels)
    plt.xticks(range(kinds_num), labels, rotation=45)
    plt.xlabel("Label")
    plt.ylabel("Pred")
    plt.title('Confusion matrix of test dataset(ATE_num:{})\n<Accuracy={:.2f}%>'.format(args.ATE_num, acc*100))
    # plt.show()
    os.makedirs("./saved/test/saved_images", exist_ok=True)

    save_path = os.path.join("./saved/test/saved_images/ATE_num{}_TestMatrix.svg".format(args.ATE_num))
    plt.savefig(save_path, dpi=600, format="svg", bbox_inches='tight')

if __name__ == "__main__":
    if args.mode == 'train_val':
        train_val_main()
    else:
        test_main()
