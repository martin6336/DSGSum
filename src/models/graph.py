import torch
import math
from torch import nn
from torch.nn import functional as F
from others.utils import tile
from models.list_encoder import list_encode
# from models.neural import MultiHeadedAttention


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class graph_encoder(nn.Module):
    def __init__(self, args, embedding):
        super().__init__()
        self.args = args
        # TODO modify
        self.renc = nn.Embedding(args.rel_num, args.graph_hdim)
        self.position_embedding = nn.Embedding(args.ent_num, args.graph_hdim)
        nn.init.xavier_normal_(self.renc.weight)
        self.gat = nn.ModuleList([Block(args) for _ in range(args.graph_layers)])
        # TODO
        self.prop = args.graph_layers
        self.list_encode = list_encode(args, embedding)
        # gate
        self.linear_gate = nn.Linear(args.graph_hdim, 1, bias=True)
        self.use_gate = args.gate
        self.gate_sig = nn.Sigmoid()

        # self.graph_actv = gelu
        self.graph_drop = nn.ModuleList([nn.Dropout(args.graph_drop) for _ in range(args.graph_layers)])
        self.graph_layer_norm = nn.ModuleList([nn.LayerNorm(args.graph_hdim, eps=1e-6) for _ in range(args.graph_layers)])
        self.embed_drop = nn.Dropout(args.graph_drop)

    def get_device(self):
        # return the device of the tensor, either "cpu"
        # or number specifiing the index of gpu.
        dev = next(self.parameters()).get_device()
        if dev == -1:
            return "cpu"
        return dev

    def graph_pad(self, tensor, length):
        return F.pad(tensor, (0, 0, 0, length - tensor.size(0)), 'constant', 0)

    def adj_pad(self, tensor, length):
        return F.pad(F.pad(tensor, (0, length - tensor.size(1), 0, 0), 'constant', 0),
                     (0, 0, 0, length - tensor.size(0)), 'constant', 1)

    def forward(self, batch, ent_embed):
        ent_list = batch.ent_list
        ent_len = batch.ent_len
        # b
        batch_len = batch.batch_len
        # b x t_p x t_p
        adj = batch.adj
        # b x t_p x h
        ent_embedding = self.list_encode(ent_list, ent_len, batch_len)
        # + position embedding
        position_ids = torch.arange(ent_embedding.size()[1], dtype=torch.long, device=ent_embedding.device)
        position_ids = position_ids.unsqueeze(0).expand((ent_embedding.size(0), ent_embedding.size(1)))
        position_emb = self.position_embedding(position_ids)
        # print(ent_embedding.size())
        # print(position_emb.size())
        # print('-------------end-----------')
        ent_embedding = ent_embedding + position_emb
        #
        ent_embedding = self.embed_drop(ent_embedding)
        N = ent_embedding.size()[1]
        # b x t_p x 1 x t_p
        mask = (adj == 0).unsqueeze(2).to(self.get_device())
        # prop 6
        for j in range(self.prop):
            # key/value: b x t_p x t_p x h
            # query: b x t_p x 1 x h
            ngraph = torch.tensor(ent_embedding.repeat(1, N, 1).view(ent_embedding.size(0), N, N, -1), requires_grad=False)
            if self.use_gate:
                # # b x t_p x 1
                # gate_mat = self.linear_gate(ent_embedding)
                # # b x t_p x 1
                # gate = self.gate_sig(torch.matmul(mask.squeeze(2).type(torch.cuda.FloatTensor), gate_mat))
                new_embedding = self.graph_drop[j](self.gat[j](ent_embedding.unsqueeze(2), ngraph, mask))
                ent_embedding = self.graph_layer_norm[j](new_embedding+ent_embedding)
            else:
                # b x t_p x h
                ent_embedding = self.graph_drop[j](self.gat[j](ent_embedding.unsqueeze(2), ngraph, mask))
        # b x t_p
        emask = torch.arange(0, N).unsqueeze(0).repeat(ent_embedding.size(0), 1).long().to(self.get_device())
        emask = ~(emask < batch_len.unsqueeze(1)).to(self.get_device())
        return ent_embedding, ~emask


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attn = MultiHeadAttention(args.graph_hdim, args.graph_hdim, args.graph_hdim, h=args.graph_encoder_head,
                                       dropout_p=args.graph_encoder_drop)
        # self.attn = MultiHeadedAttention(
        #     args.graph_encoder_head, args.graph_hdim, dropout=args.graph_encoder_drop)
        self.l1 = nn.Linear(args.graph_hdim, args.graph_hdim * 4)
        self.l2 = nn.Linear(args.graph_hdim * 4, args.graph_hdim)
        self.ln_1 = nn.LayerNorm(args.graph_hdim)
        self.ln_2 = nn.LayerNorm(args.graph_hdim)
        self.drop = nn.Dropout(args.graph_drop)
        # self.act = gelu
        self.act = nn.PReLU(args.graph_hdim * 4)
        self.gatact = nn.PReLU(args.graph_hdim)

    def forward(self, q, k, m):
        q = self.attn(q, k, mask=m).squeeze(2)
        # todo lots to modify
        # t = self.attn(q, k, k,
        #      mask=m,
        #      layer_cache=None,
        #      type="self")
        # q = t + q
        t = self.ln_1(q)
        t_2 = self.l1(t)
        batch_size = t_2.size()[0]
        ent_num = t_2.size()[1]
        hidden_dim = t_2.size()[2]
        # print('fffffffffffffffffffffffffffffffffffffffff')
        # print(t_2.size())
        q = self.drop(self.l2(self.act(t_2.view(-1, hidden_dim))))
        q = self.ln_1(q.view(batch_size, ent_num, -1) + t)
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
        self.dropout=nn.Dropout(dropout_p)
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
        #print(Q.size())
        #print(K.size())
        chunk_size = int(self._num_units / self._h)
        Q = torch.cat(Q.split(split_size=chunk_size, dim=3), dim=1)#b*aN*1*(d/a)
        K = torch.cat(K.split(split_size=chunk_size, dim=3), dim=1)#b*aN*N*(d/a)
        V = torch.cat(V.split(split_size=chunk_size, dim=3), dim=1)#aN*N*(d/a)
        # calculate QK^T
        attention = torch.matmul(Q, K.permute(0,1,3,2)) #aN*1*N
        # normalize with sqrt(dk)
        # attention and _key_dim should be in the same device.
        attention = attention / torch.sqrt(self._key_dim).to(self.get_device())

        if mask is not None:
            mask = mask.repeat(1,self._h, 1, 1) #aN*1*N
            attention.masked_fill_(mask, -1e18)
        attention = F.softmax(attention, dim=-1)
        # apply dropout
        attention = self.dropout(attention)
        # multiplyt it with V
        attention = torch.matmul(attention, V) #b*aN*1*(d/a)
        #print(attention.size())
        # convert attention back to its input original size
        restore_chunk_size = int(attention.size(1) / self._h)
        attention = torch.cat(
            attention.split(split_size=restore_chunk_size, dim=1), dim=3)
        attention += query
        return attention