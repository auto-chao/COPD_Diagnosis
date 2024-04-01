import torch
import torch.nn as nn
import os
import model.builder as builder
import copy

class Lincls(nn.Module):
    def __init__(self, args) -> None:
        super(Lincls, self).__init__()
        ATE_mdl = builder.ATE(args)
        models_save_path = "./saved/pre/saved_models/ATEnum{}_dropout{}_lr{}.pt".format(args.ATE_num, args.dropout, args.pre_slc_learning_rate)
        ATE_state = torch.load(models_save_path)
        ATE_mdl.load_state_dict(ATE_state)
        self.multi_scale = ATE_mdl.multi_scale.requires_grad_(False)
        self.multi_scale.eval()
        self.encoder = ATE_mdl.encoder.requires_grad_(False)
        self.encoder.eval()
        self.cls_head = Cls_head(args)



    def forward(self, x):
        # input.shape=[batch_size, 512, 16], out.shape = [batch_size, 10, 512, 16]
        x = self.multi_scale(x)
        # input.shape = [batch_size, 10, 512, 16], out.shape = [batch_size, 32, 1，16]
        x = self.encoder(x)
        # input.shape = [batch_size, 512, 1, 1], output.shape = [batch_size, 2]
        x = self.cls_head(x)
        return x



class Lincls_test(nn.Module):
    def __init__(self, args) -> None:
        super(Lincls_test, self).__init__()
        self.ATE_CLS = Lincls(args)
        models_save_path = "./saved/cls/saved_models/ATEnum{}_lr{}.pt".format(args.ATE_num, args.cls_learning_rate)
        ATE_state = torch.load(models_save_path)
        self.ATE_CLS.load_state_dict(ATE_state)


    def forward(self, x):
        x = self.ATE_CLS(x)
        return x


class Cls_head(nn.Module):
    def __init__(self, args) -> None:
        super(Cls_head, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_list = nn.Sequential(
            nn.Linear(args.proj_dim, 2)
        )
        # self.channel_self_attention = Channel_self_attention(args)
    def forward(self, x):
        # shape = [-1, 32, 1, 16]
        # x = self.channel_self_attention(x)
        x = self.flatten(x)
        x = self.linear_list(x)
        return x