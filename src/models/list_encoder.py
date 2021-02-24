import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class lseq_encode(nn.Module):

    def __init__(self, args, embedding):
        super().__init__()

        self.embedding = embedding
        self.input_drop = nn.Dropout(args.lstm_drop)

        self.encoder = nn.LSTM(args.enc_hidden_size, args.enc_hidden_size // 2, bidirectional=True, num_layers=args.lstm_layers, batch_first=True)

    def _cat_directions(self, h):
        h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def forward(self, inp):
        l, ilens = inp
        learned_emb = self.embedding(l)
        e = self.input_drop(learned_emb)

        sent_lens, idxs = ilens.sort(descending=True)
        # N x t_p x h
        e = e.index_select(0, idxs)
        e = pack_padded_sequence(e, sent_lens, batch_first=True)
        e, (h, c) = self.encoder(e)
        e = pad_packed_sequence(e, batch_first=True)[0]
        e = torch.zeros_like(e).scatter(0, idxs.unsqueeze(1).unsqueeze(1).expand(-1, e.size(1), e.size(2)), e)
        # hidden for seq_len(num_layer x num_direction, batch, hidden)
        h = h.transpose(0, 1)
        h = torch.zeros_like(h).scatter(0, idxs.unsqueeze(1).unsqueeze(1).expand(-1, h.size(1), h.size(2)), h)
        return e, h


class list_encode(nn.Module):
    def __init__(self, args, embedding):
        super().__init__()
        self.seqenc = lseq_encode(args, embedding)  # ,vocab=args.ent_vocab)
        # self.seqenc = lseq_encode(args,vocab=args.ent_vocab)

    def pad(self, tensor, length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(0)])

    def forward(self, batch, phlens, batch_lens, pad=True):

        # print('list_encode')
        # print(batch.size())
        # print(phlens.size())
        # print(batch_lens.size())

        batch_lens = tuple(batch_lens.tolist())
        _, enc = self.seqenc((batch, phlens))
        # num_layer:2
        enc = enc[:, 2:]
        enc = torch.cat([enc[:, i] for i in range(enc.size(1))], 1)
        m = max(batch_lens)
        encs = [self.pad(x, m) for x in enc.split(batch_lens)]
        out = torch.stack(encs, 0)
        # print(out.size())
        return out
