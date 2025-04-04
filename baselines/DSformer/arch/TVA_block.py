import torch
from torch import nn, optim
import torch.nn.functional as F


class TVA_block_att(nn.Module):
    def __init__(self,Input_len, num_id,num_layer,dropout, num_head,num_samp):
        super(TVA_block_att, self).__init__()
        self.num_lay = num_layer
        self.Time_att = Time_att(Input_len,dropout,num_head)
        self.space_att = space_att2(Input_len,num_id, dropout, num_head)
        self.cross_att = cross_att(Input_len,dropout,num_head)
        self.laynorm = nn.LayerNorm([num_id,num_samp,Input_len])
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Conv2d(in_channels=Input_len, out_channels=Input_len, kernel_size=1)
        self.linear = nn.Conv2d(in_channels=Input_len, out_channels=Input_len, kernel_size=(num_samp,1))
    def forward(self, x):

        for i in range(self.num_lay):

            x = self.cross_att(self.Time_att(x),self.space_att(x))

        x = self.linear(x.transpose(-3,-1))
        x = x.squeeze(-2)
        return x.transpose(-2,-1)


### temporal attention
class Time_att(nn.Module):
    def __init__(self, dim_input,dropout,num_head):
        super(Time_att, self).__init__()
        self.query = nn.Conv2d(in_channels=dim_input,out_channels=dim_input,kernel_size=1)
        self.key = nn.Conv2d(in_channels=dim_input,out_channels=dim_input,kernel_size=1)
        self.value = nn.Conv2d(in_channels=dim_input,out_channels=dim_input,kernel_size=1)
        self.laynorm = nn.LayerNorm([dim_input])
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(num_head,1)
    def forward(self, x):
        x = x.transpose(-3, -1)
        result = 0.0
        for i in range(self.num_head):
            q = self.dropout(self.query(x)).transpose(-3, -1)
            k = self.dropout(self.key(x)).transpose(-3, -1)
            k = k.transpose(-2, -1)
            v = self.dropout(self.value(x)).transpose(-3, -1)
            kd = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32)/self.num_head)
            line = self.dropout(self.softmax(q @ k / kd)) @ v
            if i < 1:
                result = line.unsqueeze(-1)
            else:
                result = torch.cat([result,line.unsqueeze(-1)],dim=-1)
        result = self.output(result)
        result = result.squeeze(-1)
        x = x.transpose(-3, -1) + result
        x = self.laynorm(x)
        return x


### space_attention
class space_att(nn.Module):
    def __init__(self, Input_len, dim_input,dropout,num_head):
        super(space_att, self).__init__()
        self.query = nn.Conv2d(in_channels=dim_input,out_channels=dim_input,kernel_size=1)
        self.key = nn.Conv2d(in_channels=dim_input,out_channels=dim_input,kernel_size=1)
        self.value = nn.Conv2d(in_channels=dim_input,out_channels=dim_input,kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.linear1 = nn.Linear(num_head, 1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):

        result = 0.0
        for i in range(self.num_head):
            q = self.dropout(self.query(x)).transpose(-3, -1)
            k = self.dropout(self.key(x)).transpose(-3, -1)
            k = k.transpose(-2, -1)
            v = self.dropout(self.value(x)).transpose(-3, -1)
            kd = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32)/self.num_head)

            line = self.dropout(self.softmax(q@k/kd))@ v
            if i < 1:
                result = line.unsqueeze(-1)
            else:
                result = torch.cat([result,line.unsqueeze(-1)],dim=-1)
        result = self.linear1(result)
        result = result.squeeze(-1)
        result = result.transpose(1, 3)
        return result


### space_attention2
class space_att2(nn.Module):
    def __init__(self, Input_len, dim_input,dropout,num_head):
        super(space_att2, self).__init__()
        self.query = nn.Linear(dim_input, dim_input)
        self.key = nn.Linear(dim_input, dim_input)
        self.value = nn.Linear(dim_input, dim_input)
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.linear1 = nn.Linear(num_head, 1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):

        x = x.transpose(1, 3)
        result = 0.0
        q = self.dropout(self.query(x))
        k = self.dropout(self.key(x))
        k = k.transpose(-2, -1)
        v = self.dropout(self.value(x))
        kd = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32) / self.num_head)

        for i in range(self.num_head):

            line = self.dropout(self.softmax(q@k/kd))@ v
            if i < 1:
                result = line.unsqueeze(-1)
            else:
                result = torch.cat([result,line.unsqueeze(-1)],dim=-1)
        result = self.linear1(result)
        result = result.squeeze(-1)
        result = result.transpose(1, 3)
        return result


### cross attention
class cross_att(nn.Module):
    def __init__(self, dim_input,dropout,num_head):
        super(cross_att, self).__init__()
        self.query = nn.Conv2d(in_channels=dim_input,out_channels=dim_input,kernel_size=1)
        self.key = nn.Conv2d(in_channels=dim_input,out_channels=dim_input,kernel_size=1)
        self.value = nn.Conv2d(in_channels=dim_input,out_channels=dim_input,kernel_size=1)
        self.laynorm = nn.LayerNorm([dim_input])
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(num_head,1)
    def forward(self, x, x2):
        x = x.transpose(-3, -1)
        x2 = x2.transpose(-3, -1)
        result = 0.0
        for i in range(self.num_head):
            q = self.dropout(self.query(x2)).transpose(-3, -1)
            k = self.dropout(self.key(x)).transpose(-3, -1)
            k = k.transpose(-2, -1)
            v = self.dropout(self.value(x)).transpose(-3, -1)

            kd = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32)/self.num_head)
            line = self.dropout(self.softmax(q @ k / kd)) @ v
            if i < 1:
                result = line.unsqueeze(-1)
            else:
                result = torch.cat([result,line.unsqueeze(-1)],dim=-1)
        result = self.output(result)
        result = result.squeeze(-1)
        x = x.transpose(-3, -1) + result
        x = self.laynorm(x)
        return x