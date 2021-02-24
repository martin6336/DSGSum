"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
import numpy as np

from models.encoder import PositionalEncoding
from models.neural import MultiHeadedAttention, PositionwiseFeedForward, DecoderState
# from models.graph import graph_encoder
from others.utils import tile
import torch.nn.functional as F
import math
MAX_SIZE = 5000


class TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the PositionwiseFeedForward.
      dropout (float): dropout probability(0-1.0).
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerDecoderLayer, self).__init__()


        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)

        self.context_attn_src = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.context_attn_graph = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer('mask', mask)

        self.drop_2 = nn.Dropout(dropout)
        self.drop_3 = nn.Dropout(dropout)
        self.layer_norm_3 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, inputs, memory_bank, graph_bank, src_pad_mask, tgt_pad_mask, graph_mask,
                previous_input=None, layer_cache=None, step=None):
        """
        Args:
            inputs (`FloatTensor`): `[batch_size x 1 x model_dim]`
            memory_bank (`FloatTensor`): `[batch_size x src_len x model_dim]`
            src_pad_mask (`LongTensor`): `[batch_size x 1 x src_len]`
            tgt_pad_mask (`LongTensor`): `[batch_size x 1 x 1]`

        Returns:
            (`FloatTensor`, `FloatTensor`, `FloatTensor`):

            * output `[batch_size x 1 x model_dim]`
            * attn `[batch_size x 1 x src_len]`
            * all_input `[batch_size x current_step x model_dim]`

        """

        graph_mask = graph_mask.unsqueeze(1).expand(-1, tgt_pad_mask.size()[1], -1)

        dec_mask = torch.gt(tgt_pad_mask +
                            self.mask[:, :tgt_pad_mask.size(1),
                                      :tgt_pad_mask.size(1)], 0)
        input_norm = self.layer_norm_1(inputs)
        all_input = input_norm
        if previous_input is not None:
            all_input = torch.cat((previous_input, input_norm), dim=1)
            dec_mask = None

        query = self.self_attn(all_input, all_input, input_norm,
                                     mask=dec_mask,
                                     layer_cache=layer_cache,
                                     type="self")

        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)

        src_context = self.context_attn_src(memory_bank, memory_bank, query_norm,
                                      mask=src_pad_mask,
                                      layer_cache=layer_cache,
                                      type="context")
        # initial
        if graph_bank is None:
            output = self.feed_forward(self.drop(src_context) + query)
            return output, all_input, src_context, None

        else:
        # new
            src_context = self.drop_2(src_context) + query
            src_norm = self.layer_norm_3(src_context)
            graph_context = self.context_attn_graph(graph_bank, graph_bank, src_norm,
                                          mask=graph_mask,
                                          layer_cache=layer_cache,
                                          type="graph_context")
            output = self.feed_forward(self.drop_3(graph_context) + src_context)
            return output, all_input, src_context, graph_context

    def _get_attn_subsequent_mask(self, size):
        """
        Get an attention mask to avoid using the subsequent info.

        Args:
            size: int

        Returns:
            (`LongTensor`):

            * subsequent_mask `[1 x size x size]`
        """
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask


class TransformerDecoder(nn.Module):
    """
    The Transformer decoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       dropout (float): dropout parameters
       embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
       attn_type (str): if using a seperate copy attention
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings):
        super(TransformerDecoder, self).__init__()

        # Basic attributes.
        self.decoder_type = 'transformer'
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout, self.embeddings.embedding_dim)
        #
        self.context_attn_graph = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.drop_3 = nn.Dropout(dropout)
        self.layer_norm_3 = nn.LayerNorm(d_model, eps=1e-6)
        # Build TransformerDecoder.
        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.att_weight_c = nn.Linear(self.embeddings.embedding_dim, 1)
        self.att_weight_q = nn.Linear(self.embeddings.embedding_dim, 1)
        self.att_weight_cq = nn.Linear(self.embeddings.embedding_dim, 1)
        self.graph_act = gelu
        self.graph_aware = nn.Linear(self.embeddings.embedding_dim*3, self.embeddings.embedding_dim)
        self.graph_drop = nn.Dropout(dropout)

        self.linear_filter = nn.Linear(d_model*2, 1)
        self.fix_top = torch.tensor((torch.arange(512,0,-1).type(torch.FloatTensor)/512).\
             unsqueeze(0).unsqueeze(0).expand(8, 512, -1)).to(self.get_device())
        self.fix_top.requires_grad = True
        self.fix_top = torch.nn.Parameter(self.fix_top, requires_grad=True)
        self.register_parameter("fix_top", self.fix_top)

    def get_device(self):
        # return the device of the tensor, either "cpu"
        # or number specifiing the index of gpu.
        dev = next(self.parameters()).get_device()
        if dev == -1:
            return "cpu"
        return dev

    def forward(self, tgt, memory_bank, state, memory_lengths=None,
                step=None, cache=None, memory_masks=None,
                gents=None, emask=None):
        """
        See :obj:`onmt.modules.RNNDecoderBase.forward()`
        """

        #
        src_words = state.src
        tgt_words = tgt
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()

        # Run the forward pass of the TransformerDecoder.
        # emb = self.embeddings(tgt, step=step)
        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = self.pos_emb(emb, step)

        src_memory_bank = memory_bank
        padding_idx = self.embeddings.padding_idx
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)

        if (not memory_masks is None):
            src_len = memory_masks.size(-1)
            src_pad_mask = memory_masks.expand(src_batch, tgt_len, src_len)

        else:
            # pad corespond to 1 in mask
            src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1) \
                .expand(src_batch, tgt_len, src_len)

        if state.cache is None:
            saved_inputs = []

        src_context = None
        graph_context = None
        for i in range(self.num_layers):
            prev_layer_input = None
            if state.cache is None:
                if state.previous_input is not None:
                    prev_layer_input = state.previous_layer_inputs[i]
            output, all_input, src_context, graph_context = self.transformer_layers[i](
                    output, src_memory_bank, None,
                    src_pad_mask, tgt_pad_mask, emask,
                    previous_input=prev_layer_input,
                    layer_cache=state.cache["layer_{}".format(i)]
                    if state.cache is not None else None,
                    step=step)
            if state.cache is None:
                saved_inputs.append(all_input)

        att_out = output
        ## graph information

        # b x g x e
        graph_mask = emask.unsqueeze(1).expand(-1, tgt_pad_mask.size()[1], -1)
        # #################################### filter
        # ent_sum = graph_mask.size()[-1]
        # if ent_sum >= 5:
        #     select_k = 5
        # else:
        #     select_k = ent_sum
        # # b x t_g x e/ b x e
        # score = torch.sum(torch.matmul(output, gents.transpose(1, 2)), 1)
        # # b x e
        # score_fix = self.fix_top[:ent_sum].unsqueeze(0).expand(graph_mask.size()[0], -1)
        # score = score + score_fix
        # d_model = gents.size(-1)
        # # b x k
        # top_index = torch.topk(score, select_k, -1)[1]
        # # b x k x d
        # gents_index = top_index.unsqueeze(-1).expand(-1, -1, d_model)
        # # b x k x d
        # gents_select = gents.gather(1, gents_index)
        # # b x g x k
        # mask_index = top_index.unsqueeze(1).expand(-1, graph_mask.size()[1], -1)
        # # b x g x k
        # graph_mask_select = graph_mask.gather(-1, mask_index)
        #################################
        # graph_context = self.context_attn_graph(gents, gents, output,
        #                               mask=graph_mask,
        #                               layer_cache=state.cache["layer_{}".format(0)]
        #                               if state.cache is not None else None,
        #                               type="graph_context")

        # graph_context = self.context_attn_graph(gents_select, gents_select, output,
        #                               mask=graph_mask_select,
        #                               layer_cache=state.cache["layer_{}".format(0)]
        #                               if state.cache is not None else None,
        #                               type="graph_context")
        # graph_context = self.att_flow_layer(output, gents_select, graph_mask_select)
        graph_context, attention_v = self.att_flow_layer(output, gents, graph_mask)
        output = self.feed_forward(self.drop_3(graph_context) + output)
        ##
        if state.cache is None:
            saved_inputs = torch.stack(saved_inputs)

        output = self.layer_norm(output)

        # Process the result and update the attentions.

        if state.cache is None:
            state = state.update_state(tgt, saved_inputs)
        return output, state, src_context, graph_context
        # return att_out, state, src_context, graph_context

    def init_decoder_state(self, src, memory_bank,
                           with_cache=False):
        """ Init decoder state """
        state = TransformerDecoderState(src)
        if with_cache:
            state._init_cache(memory_bank, self.num_layers)
        return state

    def att_flow_layer(self, c, q, q_mask):
        """
        :param c: (batch, c_len, hidden_size)
        :param q: (batch, q_len, hidden_size)
        :param q_mask: (batch, c_len, q_len)
        :return: (batch, c_len, q_len)
        """
        c_len = c.size(1)
        q_len = q.size(1)

        cq = []

        for i in range(q_len):
            #(batch, 1, hidden_size * 2)
            qi = q.select(1, i).unsqueeze(1)
            #(batch, c_len, 1)
            ci = self.att_weight_cq(c * qi).squeeze(-1)
            cq.append(ci)

        # (batch, c_len, q_len)
        cq = torch.stack(cq, dim=-1)
        # (batch, c_len, q_len)
        if len(cq.size()) < 3:
            cq = cq.unsqueeze(0)
            s = (self.att_weight_c(c).expand(-1, -1, q_len) + \
                 self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
                 cq).masked_fill(q_mask, 1-18)
        else:
            s = (self.att_weight_c(c).expand(-1, -1, q_len) + \
                 self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
                 cq).masked_fill(q_mask, 1-18)
        # (batch, c_len, q_len)
        a = F.softmax(s, dim=2)
        ###########################################################################
        # attention_v = a
        # attention_v = torch.bmm(c, q.permute(0,2,1))
        attention_v = cq
        ####################################################################
        # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
        c2q_att = torch.bmm(a, q)

        q2c_att = []
        for i in range(c_len):
            # (batch, 1, i)
            b_meta = F.softmax(torch.max(s[:, :(i+1), :], dim=2)[0], dim=1).unsqueeze(1)
            # (batch, 1, i) x (batch, i, h_dim)
            q2c_att_meta = torch.bmm(b_meta, c[:, :(i+1), :])
            q2c_att.append(q2c_att_meta)
        q2c_att = torch.stack(q2c_att, dim=1).squeeze(2)

        # # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
        # q2c_att = torch.bmm(b, c).squeeze()
        # # (batch, c_len, hidden_size * 2) (tiled)
        # q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
        # (batch, c_len, hidden_size * 8)

        x = torch.cat([c, c2q_att, c * c2q_att], dim=-1)
        # x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
        x = self.graph_aware(self.graph_drop(self.graph_act(x)))
        return x, attention_v


def att_flow_layer(c, q):
    """
    :param c: (batch, c_len, hidden_size * 2)
    :param q: (batch, q_len, hidden_size * 2)
    :return: (batch, c_len, q_len)
    """
    c_len = c.size(1)
    q_len = q.size(1)

    cq = []
    for i in range(q_len):
        #(batch, 1, hidden_size * 2)
        qi = q.select(1, i).unsqueeze(1)
        #(batch, c_len, 1)
        ci = self.att_weight_cq(c * qi).squeeze()
        cq.append(ci)
    # (batch, c_len, q_len)
    cq = torch.stack(cq, dim=-1)

    # (batch, c_len, q_len)
    s = self.att_weight_c(c).expand(-1, -1, q_len) + \
        self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
        cq

    # (batch, c_len, q_len)
    a = F.softmax(s, dim=2)
    # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
    c2q_att = torch.bmm(a, q)

    b=[]
    q2c_att = []
    for i in range(c_len):
        # (batch, 1, c_len)
        b_meta = F.softmax(torch.max(s[:i, :], dim=2)[0], dim=1).unsqueeze(1)
        q2c_att_meta = torch.bmm(b_meta, c[:, :i, :])
        q2c_att.append(q2c_att_meta)
    q2c_att = torch.stack(q2c_att, dim=1)
    # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
    q2c_att = torch.bmm(b, c).squeeze()
    # (batch, c_len, hidden_size * 2) (tiled)
    q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
    # (batch, c_len, hidden_size * 8)
    x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
    return x


class TransformerDecoderState(DecoderState):
    """ Transformer Decoder state base class """

    def __init__(self, src):
        """
        Args:
            src (FloatTensor): a sequence of source words tensors
                    with optional feature tensors, of size (len x batch).
        """
        self.src = src
        self.previous_input = None
        self.previous_layer_inputs = None
        self.cache = None

    @property
    def _all(self):
        """
        Contains attributes that need to be updated in self.beam_update().
        """
        if (self.previous_input is not None
                and self.previous_layer_inputs is not None):
            return (self.previous_input,
                    self.previous_layer_inputs,
                    self.src)
        else:
            return (self.src,)

    def detach(self):
        if self.previous_input is not None:
            self.previous_input = self.previous_input.detach()
        if self.previous_layer_inputs is not None:
            self.previous_layer_inputs = self.previous_layer_inputs.detach()
        self.src = self.src.detach()

    def update_state(self, new_input, previous_layer_inputs):
        state = TransformerDecoderState(self.src)
        state.previous_input = new_input
        state.previous_layer_inputs = previous_layer_inputs
        return state

    def _init_cache(self, memory_bank, num_layers):
        self.cache = {}

        for l in range(num_layers):
            layer_cache = {
                "memory_keys": None,
                "memory_values": None,
                "graph_keys": None,
                "graph_values": None
            }
            layer_cache["self_keys"] = None
            layer_cache["self_values"] = None
            self.cache["layer_{}".format(l)] = layer_cache

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.src = self.src.data.repeat(1, beam_size, 1)

    def map_batch_fn(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.src = fn(self.src, 0)
        if self.cache is not None:
            _recursive_map(self.cache)


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))




