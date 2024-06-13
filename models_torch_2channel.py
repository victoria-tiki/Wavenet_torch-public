import torch
import torch.nn as nn
import torch.nn.functional as F


class SubModule(nn.Module):
    def __init__(self, inp_shape=(4096, 1)):
        super(SubModule, self).__init__()

        self.n_filters = 32
        self.filter_width = 2
        self.dilation_rates = [2**i for i in range(11)] * 3


        self.conv1_firstit = nn.Conv1d(1,16, kernel_size=1,  padding='same')
        self.conv1_postfirstit = nn.ModuleList([nn.Conv1d(16,16, kernel_size=1,  padding='same') for dilation_rate in self.dilation_rates])[:-1]
        self.convs_f = nn.ModuleList([nn.Conv1d(16,self.n_filters, kernel_size=self.filter_width, padding='same', dilation=dilation_rate) for dilation_rate in self.dilation_rates])
        self.convs_g = nn.ModuleList([nn.Conv1d(16,self.n_filters, kernel_size=self.filter_width, padding='same', dilation=dilation_rate) for dilation_rate in self.dilation_rates])
        self.conv2 = nn.ModuleList([nn.Conv1d(self.n_filters, 16, kernel_size=1,  padding='same') for dilation_rate in self.dilation_rates])

    def forward(self, x):
        skips = []
        for i, dilation_rate in enumerate(self.dilation_rates):
            
            conv1=self.conv1_firstit if i==0 else self.conv1_postfirstit[i-1]
            
            x = F.relu(conv1(x))
            x_f = self.convs_f[i](x)
            x_g = self.convs_g[i](x)
            z = F.tanh(x_f) * F.sigmoid(x_g)
            z = F.relu(self.conv2[i](z))
            x = x + z
            skips.append(z)
        out = F.relu(torch.sum(torch.stack(skips), dim=0))
        return out
        
class PinSage(nn.Module):
    def __init__(self, dim, node_dim):
        super(PinSage, self).__init__()
        self.dim = dim
        self.node_dim=node_dim
        self.neighbor_aggregation1 = nn.Conv1d(16,dim, kernel_size=1)
        self.update_target_node = nn.Conv1d(32,self.node_dim, kernel_size=1)

    def forward(self, target_node, neighbor_1):
        neighbor_1 = F.relu(self.neighbor_aggregation1(neighbor_1)).permute(0,2,1).view(-1, 4096, self.dim, 1)
        neighbors = neighbor_1.squeeze(-1)


        out_node = torch.cat([target_node.permute(0,2,1), neighbors], dim=-1)
        out_node = F.relu(self.update_target_node(out_node.permute(0,2,1)))
        return out_node


class full_module(nn.Module):
    def __init__(self):
        super(full_module, self).__init__()
        self.sub_mod_A = SubModule()
        self.sub_mod_B = SubModule()
        #self.sub_mod_C = SubModule()
        self.dim = 16
        self.node_dim = 64
        
        self.conv1d=nn.Conv1d(self.node_dim,1, 1)
        self.pinsage_A=PinSage(self.dim,self.node_dim)
        self.pinsage_B=PinSage(self.dim,self.node_dim)

    def forward(self, x):
        x_A = x[:, :, 0].view(-1, 4096, 1).permute(0,2,1)
        x_B = x[:, :, 1].view(-1, 4096, 1).permute(0,2,1)
        x_C = x[:, :, 2].view(-1, 4096, 1).permute(0,2,1)
        
        x_A = self.sub_mod_A(x_A)
        x_B = self.sub_mod_B(x_B)

        updated_x_A = self.pinsage_A(x_A, x_B).permute(0,2,1).view(-1, 4096, self.node_dim, 1)
        updated_x_B = self.pinsage_B(x_B, x_A).permute(0,2,1).view(-1, 4096, self.node_dim, 1)

        out = torch.cat([updated_x_A, updated_x_B], dim=-1)
        out = torch.max(out, dim=-1).values#[0]

        out = self.conv1d(out.permute(0,2,1)).permute(0,2,1)
        out = F.sigmoid(out)

        return out

