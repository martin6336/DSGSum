import copy

import torch
import torch.nn as nn
from pytorch_transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_

from models.decoder import TransformerDecoder
from models.encoder import Classifier, ExtTransformerEncoder
from models.optimizers import Optimizer
from models.graph import graph_encoder
import math
import torch.nn.functional as F


def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optim'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))


    return optim

def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_bert)

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim

def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_dec)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert.model')]
    optim.set_parameters(params)
    return optim

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Generator(nn.Module):
    def __init__(self, vocab_size, dec_hidden_size, device, copy=False):
        super(Generator, self).__init__()

        self.vocab_size = vocab_size
        self.dec_hidden_size = dec_hidden_size

        self.voc_gen = nn.Sequential(nn.Linear(dec_hidden_size, vocab_size), nn.LogSoftmax(dim=-1))
        self.copy = copy
        self.copy_sigmoid = nn.Sigmoid()

        self.copy_key_linear = nn.Linear(dec_hidden_size, dec_hidden_size)
        # self.copy_ff = nn.Linear(3*dec_hidden_size, 1)
        # self.copy_query_linear = nn.Linear(3*dec_hidden_size, dec_hidden_size)
        # false but in order to read the saved model
        self.gen_linear = nn.Linear(dec_hidden_size, vocab_size)
        self.out_soft = nn.LogSoftmax(dim=-1)
        self.sigm_ff = nn.Linear(dec_hidden_size, 1)
        self.linear_score = nn.Linear(2*dec_hidden_size, 1)
        self.copy_ff = nn.Linear(2*dec_hidden_size, dec_hidden_size)
        self.copy_query_linear = nn.Linear(dec_hidden_size, dec_hidden_size)
        self.device = device
        self.norm1 = nn.LayerNorm(vocab_size)
        self.norm2 = nn.LayerNorm(vocab_size)
        self.fuse_drop = nn.Dropout(0.2)
        self.mat_drop = nn.Dropout(0.2)

        self.gen_soft = nn.Softmax(-1)
        self.copy_soft = nn.Softmax(-1)

    def forward(self, hid_vec, src, top_vec):
        if not self.copy:
            # new
            # initial
            out = self.voc_gen(hid_vec)
            return out, None
        else:
            # b x t_s
            mask_src = (src == 0)
            # b x t_g x v
            gen_dis = self.gen_linear(hid_vec)
            # gen_dis = self.gen_soft(self.gen_linear(hid_vec))

            # b x t_g x t_s
            # score = torch.matmul(fuse_pro, src_pro.transpose(1, 2))
            dec_num = hid_vec.size()[1]
            src_num = src.size()[1]
            # b x t_g x t_s x 2*h
            score_mat = torch.cat((hid_vec.unsqueeze(2).expand((-1,-1,src_num,-1)),
                                   top_vec.unsqueeze(1).expand((-1,dec_num,-1,-1))), -1)
            # b x t_g x t_s
            score = self.linear_score(score_mat).squeeze(-1)
            # score = torch.matmul(hid_vec, top_vec.transpose(1, 2))
            score = score.masked_fill_(mask_src.unsqueeze(1), -1e-8)

            # score = self.copy_soft(score)
            copy_dis = torch.zeros(gen_dis.size()).to(self.device).scatter_(2, src.unsqueeze(1).expand_as(score), score)
            # gen_dis = self.norm1(gen_dis)
            # copy_dis = self.norm2(copy_dis)
            src_context = torch.matmul(score, top_vec)
            # b x t_g x h
            # fuse_mat = gelu(self.copy_ff(self.fuse_drop(torch.cat((hid_vec, src_context), -1))))
            fuse_mat = self.copy_ff(torch.cat((hid_vec, src_context), -1))
            # b x t_g x 1
            copy = self.copy_sigmoid(self.sigm_ff(fuse_mat))
            out = (1-copy)*gen_dis + copy*copy_dis
            out = self.out_soft(out)
            # print(copy)
            return out, copy


def get_generator(vocab_size, dec_hidden_size, device, copy):
    generator = Generator(vocab_size, dec_hidden_size, device, copy)
    # gen_func = nn.LogSoftmax(dim=-1)
    # generator=nn.Sequential(nn.Linear(dec_hidden_size, vocab_size), gen_func)
    # generator.to(device)
    return generator


class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if(large):
            # self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
            self.model = BertModel.from_pretrained(temp_dir, cache_dir='~/.cache/torch/pytorch_transformers')

        else:
            # self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)
            self.model = BertModel.from_pretrained(temp_dir, cache_dir='~/.cache/torch/pytorch_transformers')

        self.finetune = finetune

    def forward(self, x, segs, mask):
        if(self.finetune):
            top_vec, _ = self.model(x, segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, segs, attention_mask=mask)
        return top_vec


class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        self.ext_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
                                               args.ext_dropout, args.ext_layers)
        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.ext_hidden_size,
                                     num_hidden_layers=args.ext_layers, num_attention_heads=args.ext_heads, intermediate_size=args.ext_ff_size)
            self.bert.model = BertModel(bert_config)
            self.ext_layer = Classifier(self.bert.model.config.hidden_size)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

        self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls


class AbsSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint=None, bert_from_extractive=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        if bert_from_extractive is not None:
            self.bert.model.load_state_dict(
                dict([(n[11:], p) for n, p in bert_from_extractive.items() if n.startswith('bert.model')]), strict=True)

        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.enc_hidden_size,
                                     num_hidden_layers=args.enc_layers, num_attention_heads=8,
                                     intermediate_size=args.enc_ff_size,
                                     hidden_dropout_prob=args.enc_dropout,
                                     attention_probs_dropout_prob=args.enc_dropout)
            self.bert.model = BertModel(bert_config)
        self.graph_encoder = graph_encoder(args, self.bert.model.embeddings)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
        if (self.args.share_emb):
            tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)

        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)
        #
        for name, param in self.decoder.named_parameters():
            if name == 'fix_top':
                xavier_uniform_(param)
        #
        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device, args.copy)
        self.generator.voc_gen[0].weight = self.decoder.embeddings.weight

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            if(args.use_bert_emb):
                tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
                self.decoder.embeddings = tgt_embeddings
                self.generator.voc_gen[0].weight = self.decoder.embeddings.weight
        self.copy = args.copy
        self.to(device)

    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls, batch=None):
        #
        gents, emask = self.graph_encoder(batch, self.bert.model.embeddings)
        #
        top_vec = self.bert(src, segs, mask_src)
        ent_top_vec = None
        if self.copy == True:
            ent_top_vec = self.bert(batch.ent_src, batch.ent_seg_ids, batch.mask_ent_src)
            # sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
            # sents_vec = sents_vec * mask_cls[:, :, None].float()

        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, state, src_context, graph_context = self.decoder(tgt[:, :-1], top_vec, dec_state,
                                                                          gents=gents, emask=emask)
        return decoder_outputs, None, src_context, graph_context, top_vec, ent_top_vec, emask
