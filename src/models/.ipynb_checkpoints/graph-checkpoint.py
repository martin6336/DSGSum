import torch
import math
from torch import nn
from torch.nn import functional as F
from others.utils import tile


class graph_encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # TODO modify
        self.renc = nn.Embedding(args.rel_num, args.graph_hdim)
        nn.init.xavier_normal_(self.renc.weight)
        self.gat = nn.ModuleList([Block(args) for _ in range(args.graph_layers)])
        # TODO
        self.prop = args.graph_layers
        self.root_index = args.rel_num-1

    def get_device(self):
        # return the device of the tensor, either "cpu"
        # or number specifiing the index of gpu.
        dev = next(self.parameters()).get_device()
        if dev == -1:
            return "cpu"
        return dev

    def pad(self, tensor, length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(0)])

    def forward(self, ents, rels, adjs, ent_embed):
        # rels/ents: index list of rels/ents (not padded)
        # adjs: list of adjancent matrix
          #print(ents[0].size())
        # print(adjs[0].size())
        # print(rels[0].size())

        # print(ents[0])
        # print(root.size())

        graphs = []
        for i, adj in enumerate(adjs): #这里看能不能用batch

            # (ent+rel) x h
            if len(ents[i]) > 0:
                venc = ent_embed(ents[i].unsqueeze(0)).squeeze(0)
                vrel = self.renc(rels[i].unsqueeze(0)).squeeze(0)
                root = self.renc(torch.tensor(46).type(torch.long).to(self.get_device())).unsqueeze(0)

                vgraph = torch.cat((venc, vrel, root), 0)
                # vgraph = torch.cat((venc, vrel), 0)

            else:
                vgraph = self.renc(rels[i].unsqueeze(0)).squeeze(0)

            N = vgraph.size(0)
            mask = (adj == 0).unsqueeze(1) #不相邻的点 N*1*N
            # prop 6
            for j in range(self.prop):
                # Ni x Ni x h
                ngraph = torch.tensor(vgraph.repeat(N, 1).view(N, N, -1), requires_grad=False) #N*N*d
                vgraph = self.gat[j](vgraph.unsqueeze(1), ngraph, mask) #N*1*d

            graphs.append(vgraph)

        # graph : list of size b, each of which is Ni x h
        elens = [x.size(0) for x in graphs]
        gents = [self.pad(x, max(elens)) for x in graphs]
        gents = torch.stack(gents, 0)
        elens = torch.LongTensor(elens)
        emask = torch.arange(0, gents.size(1)).unsqueeze(0).repeat(gents.size(0), 1).long()
        # emask and vents should be in the same device.
        # torch.Size([2, 10])
        # TODO should be < not <=
        emask = (emask < elens.unsqueeze(1)).to(self.get_device())

        return gents, emask


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attn = MultiHeadAttention(args.graph_hdim, args.graph_hdim, args.graph_hdim, h=4, dropout_p=args.graph_drop)
        self.l1 = nn.Linear(args.graph_hdim, args.graph_hdim * 4)
        self.l2 = nn.Linear(args.graph_hdim * 4, args.graph_hdim)
        self.ln_1 = nn.LayerNorm(args.graph_hdim)
        self.ln_2 = nn.LayerNorm(args.graph_hdim)
        self.drop = nn.Dropout(args.graph_drop)
        # self.act = gelu
        self.act = nn.PReLU(args.graph_hdim * 4)
        self.gatact = nn.PReLU(args.graph_hdim)

    def forward(self, q, k, m):
        q = self.attn(q, k, mask=m).squeeze(1)
        t = self.ln_1(q)
        q = self.drop(self.l2(self.act(self.l1(t))))
        q = self.ln_1(q + t)
        return q


class MultiHeadAttention(nn.Module):
    def __init__(self, query_dim, key_dim, num_units,
                 dropout_p=0.5, h=8, is_masked=False):
        super(MultiHeadAttention, self).__init__()

        if query_dim != key_dim:
            raise ValueError("query_dim and key_dim must be the same")
        if num_units % h != 0:
            raise ValueError("num_units must be dividable by h")
        if query_dim != num_units:
            raise ValueError("to employ residual connection, the number of "
                             "query_dim and num_units must be the same")

        self._num_units = num_units
        self._h = h
        self._key_dim = torch.tensor(key_dim, requires_grad=False).float()
        self._dropout_p = dropout_p
        self._is_masked = is_masked

        self.query_layer = nn.Linear(query_dim, num_units, bias=False)
        self.key_layer = nn.Linear(key_dim, num_units, bias=False)
        self.value_layer = nn.Linear(key_dim, num_units, bias=False)
        self.bn = nn.BatchNorm1d(num_units)
        self.ln = nn.LayerNorm(num_units)
        self.dropout = nn.Dropout(dropout_p)

    def get_device(self):
        # return the device of the tensor, either "cpu"
        # or number specifiing the index of gpu.
        dev = next(self.parameters()).get_device()
        if dev == -1:
            return "cpu"
        return dev

    def forward(self, query, keys, mask=None):
        Q = self.query_layer(query) #N*1*d
        K = self.key_layer(keys) #N*N*d
        V = self.value_layer(keys) #N*N*d

        # split each Q, K and V into h different values from dim 2
        # and then merge them back together in dim 0
        chunk_size = int(self._num_units / self._h)
        Q = torch.cat(Q.split(split_size=chunk_size, dim=2), dim=0)#aN*1*(d/a)
        K = torch.cat(K.split(split_size=chunk_size, dim=2), dim=0)#aN*N*(d/a)
        V = torch.cat(V.split(split_size=chunk_size, dim=2), dim=0)

        # calculate QK^T
        attention = torch.matmul(Q, K.transpose(1, 2)) #aN*1*N
        # normalize with sqrt(dk)

        # attention and _key_dim should be in the same device.
        attention = attention / torch.sqrt(self._key_dim).to(self.get_device())

        if mask is not None:
            mask = mask.repeat(self._h, 1, 1) #aN*1*N
            attention.masked_fill_(mask, -float('inf'))
        attention = F.softmax(attention, dim=-1)
        # apply dropout
        # attention = self.dropout(attention)
        attention = F.dropout(attention, self._dropout_p, training=False)
        # multiplyt it with V

        attention = torch.matmul(attention, V)
        # convert attention back to its input original size
        restore_chunk_size = int(attention.size(0) / self._h)
        attention = torch.cat(
            attention.split(split_size=restore_chunk_size, dim=0), dim=2)
        attention += query
        return attention



#####################################################################################################################
#
#
# class graph_encoder(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         # TODO modify
#         self.renc = nn.Embedding(args.rel_num, args.graph_hdim)
#         # 已经加过1
#         self.root_index = args.rel_num
#         nn.init.xavier_normal_(self.renc.weight)
#         self.gat = nn.ModuleList([Block(args) for _ in range(args.graph_layers)])
#         # TODO
#         self.prop = args.graph_layers
#     def get_device(self):
#         # return the device of the tensor, either "cpu"
#         # or number specifiing the index of gpu.
#         dev = next(self.parameters()).get_device()
#         if dev == -1:
#             return "cpu"
#         return dev
#
#     def graph_pad(self, tensor, length):
#         return F.pad(tensor,(0,0,0,length-tensor.size(0)),'constant',0)
#     def adj_pad(self,tensor,length):
#         return F.pad(F.pad(tensor,(0,length-tensor.size(1),0,0),'constant',0),(0,0,0,length-tensor.size(0)),'constant',1)
#     def forward(self, ents, rels, adjs, ent_embed):
#         # rels/ents: index list of rels/ents (not padded)
#         # adjs: list of adjancent matrix
#           #print(ents[0].size())
#         # print(adjs[0].size())
#         # print(rels[0].size())
#         print(self.get_device())
#         # print(ents[0])
#         graphs = []
#         for i, adj in enumerate(adjs): #这里看能不能用batch
#
#             # (ent+rel) x h
#             if len(ents[i]) > 0:
#                 venc = ent_embed(ents[i].unsqueeze(0)).squeeze(0)
#                 vrel = self.renc(rels[i].unsqueeze(0)).squeeze(0)
#                 ##
#                 root = self.renc(torch.tensor(self.root_index).to(self.get_device())).unsqueeze(0)
#                 ##
#                 vgraph = torch.cat((venc, vrel, root), 0)
#             else:
#                 vgraph = self.renc(rels[i].unsqueeze(0)).squeeze(0)
#             #N*d
#             graphs.append(vgraph)
#         # graph : list of size b, each of which is Ni x h
#         elens = [x.size(0) for x in graphs]
#         N=max(elens)
#         graph1 = [self.graph_pad(x, N) for x in graphs]
#         adj1=[self.adj_pad(x,N) for x in adjs]
#         graph1=torch.stack(graph1,0)
#         adj1=torch.stack(adj1,0)
#         mask = (adj1 == 0).unsqueeze(2) #不相邻的点 b*N*1*N,把padding的要摁上
#         # prop 6
#         for j in range(self.prop):
#                 # Ni x Ni x h
#             ngraph = torch.tensor(graph1.repeat(1, N, 1).view(graph1.size(0),N, N, -1), requires_grad=False) #N*N*d
#             graph1 = self.gat[j](graph1.unsqueeze(2), ngraph, mask) #N*1*d
#             #print(graph1) #补零的样本出了问题
#         elens = torch.LongTensor(elens)
#         emask = torch.arange(0, graph1.size(1)).unsqueeze(0).repeat(graph1.size(0), 1).long()
#         # emask and vents should be in the same device.
#         # torch.Size([2, 10])
#         emask = (emask <= elens.unsqueeze(1)).to(self.get_device())
#         return graph1, emask
#
#
# def gelu(x):
#     return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
#
#
# class Block(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.attn = MultiHeadAttention(args.graph_hdim, args.graph_hdim, args.graph_hdim, h=4, dropout_p=args.graph_drop)
#         self.l1 = nn.Linear(args.graph_hdim, args.graph_hdim * 4)
#         self.l2 = nn.Linear(args.graph_hdim * 4, args.graph_hdim)
#         self.ln_1 = nn.LayerNorm(args.graph_hdim)
#         self.ln_2 = nn.LayerNorm(args.graph_hdim)
#         self.drop = nn.Dropout(args.graph_drop)
#         # self.act = gelu
#         self.act = nn.PReLU(args.graph_hdim * 4)
#         self.gatact = nn.PReLU(args.graph_hdim)
#
#     def forward(self, q, k, m):
#         #print(q)
#         q = self.attn(q, k, mask=m).squeeze(2)
#         #print(q)
#         batch_size=q.size(0)
#         N=q.size(1)
#         q=q.view(batch_size*N,-1)
#         t = self.ln_1(q)
#         q = self.drop(self.l2(self.act(self.l1(t))))
#         q = self.ln_1(q + t).view(batch_size,N,-1)
#         return q
#
#
# class MultiHeadAttention(nn.Module):
#     def __init__(self, query_dim, key_dim, num_units,
#                  dropout_p=0.5, h=8, is_masked=False):
#         super(MultiHeadAttention, self).__init__()
#
#         if query_dim != key_dim:
#             raise ValueError("query_dim and key_dim must be the same")
#         if num_units % h != 0:
#             raise ValueError("num_units must be dividable by h")
#         if query_dim != num_units:
#             raise ValueError("to employ residual connection, the number of "
#                              "query_dim and num_units must be the same")
#
#         self._num_units = num_units
#         self._h = h
#         self._key_dim = torch.tensor(key_dim, requires_grad=False).float()
#         self._dropout_p = dropout_p
#         self._is_masked = is_masked
#
#         self.query_layer = nn.Linear(query_dim, num_units, bias=False)
#         self.key_layer = nn.Linear(key_dim, num_units, bias=False)
#         self.value_layer = nn.Linear(key_dim, num_units, bias=False)
#         self.bn = nn.BatchNorm1d(num_units)
#         self.ln = nn.LayerNorm(num_units)
#         self.dropout=nn.Dropout(dropout_p)
#     def get_device(self):
#         # return the device of the tensor, either "cpu"
#         # or number specifiing the index of gpu.
#         dev = next(self.parameters()).get_device()
#         if dev == -1:
#             return "cpu"
#         return dev
#
#     def forward(self, query, keys, mask=None):
#         Q = self.query_layer(query) #N*1*d
#         K = self.key_layer(keys) #N*N*d
#         V = self.value_layer(keys) #N*N*d
#         # split each Q, K and V into h different values from dim 2
#         # and then merge them back together in dim 0
#         #print(Q.size())
#         #print(K.size())
#         chunk_size = int(self._num_units / self._h)
#         Q = torch.cat(Q.split(split_size=chunk_size, dim=3), dim=1)#b*aN*1*(d/a)
#         K = torch.cat(K.split(split_size=chunk_size, dim=3), dim=1)#b*aN*N*(d/a)
#         V = torch.cat(V.split(split_size=chunk_size, dim=3), dim=1)#aN*N*(d/a)
#         # calculate QK^T
#         attention = torch.matmul(Q, K.permute(0,1,3,2)) #aN*1*N
#         # normalize with sqrt(dk)
#         # attention and _key_dim should be in the same device.
#         attention = attention / torch.sqrt(self._key_dim).to(self.get_device())
#
#         if mask is not None:
#             mask = mask.repeat(1,self._h, 1, 1) #aN*1*N
#             attention.masked_fill_(mask, -float('inf'))
#         attention = F.softmax(attention, dim=-1)
#         # apply dropout
#         attention = self.dropout(attention)
#         # multiplyt it with V
#         attention = torch.matmul(attention, V) #b*aN*1*(d/a)
#         #print(attention.size())
#         # convert attention back to its input original size
#         restore_chunk_size = int(attention.size(1) / self._h)
#         attention = torch.cat(
#             attention.split(split_size=restore_chunk_size, dim=1), dim=3)
#         attention += query
#         return attention
