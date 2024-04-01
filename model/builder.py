import torch
import torch.nn as nn
import copy
import torch.nn.functional as F

class ATE(nn.Module):
    def __init__(self, args) -> None:
        '''
        proj_dim: the dim of the projector's output
        pred_dim: the dim of the predictor's output
        '''
        super(ATE, self).__init__()
        proj_dim = args.proj_dim
        pred_dim = args.pred_dim
        ATE_num = args.ATE_num
        dropout = args.dropout
        head = args.head
        temporal_dim = args.temporal_dim
        feature_dim = args.feature_dim
        d_ff = args.d_ff
        pretext_dim = args.pretext_dim
        # input.shape = [batch_size, 512, 16], out.shape = [batch_size, 10, 512, 16]
        self.multi_scale = Multi_scale()
        # input.shape = [batch_size, 10, 512, 16], out.shape = [batch_size, 32, 1, 16]
        self.sublayer = Sublayer_connection()
        self.ATE_layer = ATE_layer(head, dropout, temporal_dim, feature_dim, d_ff, self.sublayer)
        self.encoder = Encoder(ATE_num, self.ATE_layer, args)
        # input.shape = [batch_size, 32, 1, 16], output.shape = [batch_size, 2048]
        self.projection_head = Projection_head(proj_dim, pretext_dim)
        # input.shape = [batch_size, 2048], output.shape = [batch_size, 2048]
        self.predictor = Predictor(pred_dim, pretext_dim)
        
    def forward(self, x1, x2):
        x1_scale = self.multi_scale(x1)
        x2_scale = self.multi_scale(x2)
        z1 = self.encoder(x1_scale)
        z2 = self.encoder(x2_scale)

        z1 = self.projection_head(z1)
        z2 = self.projection_head(z2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()


class Multi_scale(nn.Module):
    def __init__(self) -> None:
        super(Multi_scale, self).__init__()
        self.conv_list = nn.ModuleList([
        nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(1,1),stride=(1,1),padding=(0,0),bias=False),
        nn.Conv2d(in_channels=1,out_channels=2,kernel_size=(1,3),stride=(1,1),padding=(0,1),bias=False),
        nn.Conv2d(in_channels=1,out_channels=2,kernel_size=(1,5),stride=(1,1),padding=(0,2),bias=False),
        nn.Conv2d(in_channels=1,out_channels=2,kernel_size=(1,7),stride=(1,1),padding=(0,3),bias=False),
        nn.Conv2d(in_channels=1,out_channels=2,kernel_size=(1,9),stride=(1,1),padding=(0,4),bias=False)            
        ])

    def forward(self, x):
        # x.shape = [batch_size,512,16]
        x = x.transpose(1,2).unsqueeze(1) # shape = [batch_size,1,16,512] 
        output = torch.concat([x, self.conv_list[0](x)], dim=1)
        for i in range(1,5):
            output = torch.concat([output, self.conv_list[i](x)], dim=1)
        output = output.transpose(2,3)
        # output.shape = [batch_size, 10, 512, 16]
        return output


class Encoder(nn.Module):
    def __init__(self, ATE_num, ATE_layer, args) -> None:
        super(Encoder, self).__init__()
        # input.shape = [batch_size,10,512,16], output.shape = [batch_size,32,1,16]
        self.conv_list = nn.ModuleList([
            nn.Conv2d(in_channels=10,out_channels=args.channel_dim, kernel_size=(512,1),stride=(1,1),padding=(0,0),bias=False)
        ])
        self.ATE_list = clones(ATE_layer, ATE_num)
        
    def forward(self, x):
        for ate in self.ATE_list:
            x = ate(x)
        for conv in self.conv_list:
            x = conv(x)
        return x




class Predictor(nn.Module):
    def __init__(self, pred_dim, pretext_dim) -> None:
        super(Predictor,self).__init__()
        self.pred_module = nn.Sequential(
            nn.Linear(pretext_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, pretext_dim)
        )

    def forward(self, x):
        x = self.pred_module(x)
        return x


class Projection_head(nn.Module):
    def __init__(self, proj_dim, pretext_dim) -> None:
        super(Projection_head, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(proj_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(32, pretext_dim)
        )
        self.module_list = nn.ModuleList([
            nn.Linear(proj_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True), # first layer
            nn.Linear(proj_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True), # second layer
            self.fc ,
            nn.BatchNorm1d(pretext_dim, affine=False) # output layer
        ])
        
    def forward(self, x):
        # shape = [batch_size, 32, 1, 16]
        x = self.flatten(x)
        # shape = [batch_size, 512]
        for module in self.module_list:
            x = module(x)
        # shape = [batch_size, 2048]
        return x

class ATE_layer(nn.Module):
    def __init__(self, head, dropout, temporal_dim, feature_dim, d_ff, sublayer) -> None:
        super(ATE_layer, self).__init__()
        self.sublayer = sublayer
        self.ff = nn.Sequential(
            ATE_layer_att(head, dropout, temporal_dim, feature_dim, d_ff),
            ATE_layer_fw(head,dropout, temporal_dim, feature_dim, d_ff)
        )
    
    def forward(self, x):
        x = self.sublayer(x, lambda x: self.ff(x))
        return x

# input.shape = [batch_size, 10, 512, 16], out.shape = [batch_size, 10, 512, 16]
class ATE_layer_att(nn.Module):
    # include the temporal_attention and feature_attention
    def __init__(self, head, dropout, temporal_dim, feature_dim, d_ff) -> None:
        super(ATE_layer_att, self).__init__()
        self.temporal_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=head, dropout=dropout)
        self.feature_attention = nn.MultiheadAttention(embed_dim=temporal_dim, num_heads=head, dropout=dropout)
        self.linears_1 = clones(nn.Linear(feature_dim,feature_dim), 4)
        self.linears_2 = clones(nn.Linear(temporal_dim,temporal_dim), 4)
        self.flatten = nn.Flatten(0,1)
        self.pre_layer_norm = nn.LayerNorm([temporal_dim,feature_dim])


    def forward(self, x):
        # ATE
        x = self.flatten(x)
        # x.shape = [batch_size*10, 512, 16]
        x = self.pre_layer_norm(x)
        q1, k1, v1 = [linear(input)  for linear, input in zip(self.linears_1, (x,x,x))] 
        x1, _ = self.temporal_attention(q1,k1,v1)
        x1 = self.linears_1[-1](x1)
        # x.shape = [batch_size*10, 16, 512]
        x = x.transpose(1,2)
        q2, k2, v2 = [linear(input)  for linear, input in zip(self.linears_2, (x,x,x))]
        x2, _ = self.feature_attention(q2,k2,v2)
        x2 = self.linears_2[-1](x2).transpose(1,2)
        # shape = [b_z*10, 512, 16]
        x = x1 + x2
        x = x.reshape(-1, 10, 512, 16)
        # x.shape = [batch_size, 10, 512, 16]
        return x

class ATE_layer_fw(nn.Module):
    def __init__(self, head, dropout, temporal_dim, feature_dim, d_ff) -> None:
        super(ATE_layer_fw, self).__init__()
        # input.shape = [batch_size, 10, 512, 16] = output.shape
        self.flatten = nn.Flatten(0,1)
        self.fw = Feedforward(feature_dim, d_ff, dropout)
        self.pre_layer_norm = nn.LayerNorm([temporal_dim,feature_dim])

    def forward(self, x):
        x = self.flatten(x)
        x = self.pre_layer_norm(x)
        x = self.fw(x)
        x = x.reshape(-1, 10, 512, 16)
        return x

class Feedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout) -> None:
        super(Feedforward, self).__init__()
        self.d_ff = d_ff
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff,d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.w2(self.dropout1(F.relu(self.w1(x)))))


class Sublayer_connection(nn.Module):
    def __init__(self) -> None:
        super(Sublayer_connection, self).__init__()
    
    def forward(self, x, sublayer):
        
        return x + sublayer(x)

def clones(module, N):
    '''
    Generate the clone function of the same network layer
    module: the target network layer to clone
    N: the number of clones needed# In the function
    we deep copy the modules N times through the for loop, 
    making each module a separate layer
    Then put it in a list of type nn.ModuleList
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
