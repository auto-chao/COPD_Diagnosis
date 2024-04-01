import math

def adjust_learning_rate(optimizer, init_lr, epoch, args):
    # warm_epochs = 20
    # if epoch <= warm_epochs:
    #     cur_lr = init_lr * 0.1 * (2. - math.cos(math.pi * epoch / warm_epochs))
    # else:
    #     cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * (epoch-warm_epochs) / (args.epochs-warm_epochs)))
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr