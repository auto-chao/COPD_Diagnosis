import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def plot_image(train_loss_array, valid_loss_array, args):
    epochs = args.epochs
    plt.figure()
    x = np.arange(epochs)
    plt.plot(x, train_loss_array, 'b-')
    plt.plot(x, valid_loss_array, 'r-')
    plt.title('Training and validation loss of classification\n(ATE_num:{}; Learning_rate:{})'.format(args.ATE_num, args.learning_rate))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["train_loss", 'val_loss'], bbox_to_anchor=(1, 1), loc="upper right")
    # plt.show()
    os.makedirs("./saved/pre/saved_images", exist_ok=True)
    save_path = "./saved/pre/saved_images/ATEnum{}_dropout{}_lr{}.svg".format(args.ATE_num, args.dropout, args.learning_rate)
    plt.savefig(save_path, dpi=600, format="svg", bbox_inches='tight')


def plot_image_cls(train_loss_array, valid_loss_array, train_acc_array, valid_acc_array, args):
    epochs = args.epochs
    plt.figure(figsize=(20, 6))
    x = np.arange(epochs)
    plt.subplot(121)
    x = np.arange(epochs)
    plt.plot(x, train_loss_array, 'b-')
    plt.plot(x, valid_loss_array, 'r-')
    plt.title('Training and validation loss of classification\n(ATE_num:{}; Learning_rate:{})'.format(args.ATE_num, args.learning_rate))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["train_loss", 'val_loss'], bbox_to_anchor=(1, 1), loc="upper right")

    plt.subplot(122)
    x = np.arange(epochs)
    plt.plot(x, train_acc_array, 'b-')
    plt.plot(x, valid_acc_array, 'r-')
    plt.title('Training and validation accuracy of classification\n(ATE_num:{}; Learning_rate:{})'.format(args.ATE_num, args.learning_rate))
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["train_acc", 'val_acc'], bbox_to_anchor=(1, 0), loc="lower right")

    # plt.show()
    os.makedirs("./saved/cls/saved_images", exist_ok=True)
    save_path = "./saved/cls/saved_images/ATEnum{}_lr{}.svg".format(args.ATE_num, args.learning_rate)
    plt.savefig(save_path, dpi=1200, format="svg", bbox_inches='tight')
