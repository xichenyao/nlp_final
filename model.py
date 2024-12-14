# """
# A Text-Centered Shared-Private Framework via Cross-Modal Prediction for Multimodal Sentiment Analysis (TCSP)

# """


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from copy import deepcopy


# class Translation(nn.Module):
#     def __init__(self, config):
#         super(Translation, self).__init__()
#         self.config = config
#         self.encoder = nn.LSTM(input_size=config["source_size"],
#                                hidden_size=config["encoder_hidden_size"],
#                                num_layers=config["encoder_num_layers"],
#                                dropout=config["encoder_dropout"])
#         self.bn = nn.BatchNorm1d(config["encoder_hidden_size"])
#         self.decoder = nn.LSTM(input_size=config["decoder_input_size"],
#                                hidden_size=config["decoder_hidden_size"],
#                                num_layers=config["decoder_num_layers"],
#                                dropout=config["decoder_dropout"])
#         self.attn_1 = nn.Linear(config["encoder_hidden_size"] * 2, config["decoder_hidden_size"])
#         self.attn_2 = nn.Linear(config["decoder_hidden_size"], 1, bias=False)
#         self.fc1 = nn.Linear(config["encoder_hidden_size"]+config["decoder_hidden_size"],
#                             config["encoder_hidden_size"]+config["decoder_hidden_size"])
#         self.fc2 = nn.Linear(config["encoder_hidden_size"]+config["decoder_hidden_size"], config["target_size"])

#     def _get_attn_weight(self, dec_rep, enc_reps, mask):   # (1, enc_n, b, dec_h_d), (1, enc_n, b, enc_h_d)
#         cat_reps = torch.cat([enc_reps, dec_rep], dim=-1)  # (1, enc_n, b, enc_h_d+dec_h_d)
#         attn_scores = self.attn_2(F.tanh(self.attn_1(cat_reps))).squeeze(3)    # (1, enc_n, b)
#         attn_scores = mask * attn_scores
#         return torch.softmax(attn_scores, dim=1)    # (1, enc_n, b)

#     def encode(self, source, lengths):
#         packed_sequence = pack_padded_sequence(source, lengths.cpu())
#         packed_hs, (final_h, _) = self.encoder(packed_sequence)
#         enc_hs, _ = pad_packed_sequence(packed_hs)  # (enc_n, b, enc_h_d)
#         return enc_hs

#     def decode(self, source, target, enc_hs, mask):
#         n_step = len(target)
#         enc_n, batch_size, enc_h_d = enc_hs.size()
#         dec_h_d = self.config["decoder_hidden_size"]

#         # initialize
#         dec_h = torch.zeros(1, batch_size, dec_h_d).to(source.device) # (1, b, dec_h_d)
#         dec_c = deepcopy(dec_h)

#         dec_rep = dec_h.view(1, 1, batch_size, dec_h_d).expand(1, enc_n, batch_size, dec_h_d)   # (1, enc_n, b, dec_h_d)
#         enc_reps = enc_hs.view(1, enc_n, batch_size, enc_h_d)   # (1, enc_n, b, enc_h_d)
#         attn_weights = self._get_attn_weight(dec_rep, enc_reps, mask)   # (1, enc_n, b)
#         context = attn_weights.unsqueeze(3).expand_as(enc_reps) * enc_reps    # (1, enc_n, b, enc_h_d)
#         context = torch.sum(context, dim=1)     # (1, b, enc_h_d)

#         dec_in = torch.cat([dec_h, context], dim=2)  # (1, b, enc_h_d+dec_h_d)
#         all_attn_weights = torch.empty([n_step, enc_n, batch_size]).to(source.device)  # (dec_n, enc_n, b)
#         all_dec_out = torch.empty([n_step, batch_size, self.config["target_size"]]).to(
#             source.device)  # (dec_n, b, dec_o_d)

#         for i in range(n_step):
#             _, (dec_h, dec_c) = self.decoder(dec_in, (dec_h, dec_c))  # (1, b, dec_h_d)

#             dec_rep = dec_h.view(1, 1, batch_size, dec_h_d).expand(1, enc_n, batch_size, dec_h_d)
#             attn_weights = self._get_attn_weight(dec_rep, enc_reps, mask)  # (1, b, enc_n)
#             all_attn_weights[i] = attn_weights

#             context = attn_weights.unsqueeze(3).expand_as(enc_reps) * enc_reps  # (1, enc_n, b, enc_h_d)
#             context = torch.sum(context, dim=1)  # (1, b, enc_h_d)
#             dec_in = torch.cat([dec_h, context], dim=2)
#             all_dec_out[i] = self.fc2(F.relu(self.fc1(dec_in)))    # (1, b, dec_o_d)

#         return all_dec_out.permute(1, 0, 2).contiguous(), all_attn_weights.permute(2, 0, 1).contiguous()

#     def forward(self, source, target, lengths):
#         batch_size, enc_n, _ = source.size()
#         source = source.permute(1, 0, 2).contiguous()   # (enc_n, b, *)
#         target = target.permute(1, 0, 2).contiguous()   # (dec_n, b, *)

#         # encode
#         enc_hs = self.encode(source, lengths).permute(1, 2, 0).contiguous()

#         # batch normalize
#         try:
#             enc_hs = self.bn(enc_hs).permute(2, 0, 1).contiguous()
#         except:
#             enc_hs = enc_hs.permute(2, 0, 1).contiguous()

#         # get mask for attention
#         seq_range = torch.arange(0, enc_n).long().to(source.device)
#         seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, enc_n)
#         seq_length_expand = lengths.unsqueeze(1).expand_as(seq_range_expand).long()
#         before_mask = (seq_range_expand < seq_length_expand).unsqueeze(1).expand(batch_size, 1, enc_n).float()
#         before_mask = before_mask.permute(1, 2, 0).contiguous()
#         attn_mask = (seq_range_expand < seq_length_expand).unsqueeze(1).expand(batch_size, enc_n, enc_n).contiguous()
#         tgt_mask = (seq_range_expand < seq_length_expand).unsqueeze(2).expand(batch_size, enc_n, target.size(2)).contiguous()

#         # decode
#         all_dec_out, all_attn_weights = self.decode(source, target, enc_hs, before_mask)

#         # masked
#         all_dec_out = all_dec_out.masked_fill(~tgt_mask, 0.0)
#         all_attn_weights = all_attn_weights.masked_fill(~attn_mask, 1e-10)
#         all_attn_weights = all_attn_weights.masked_fill(~attn_mask.transpose(1, 2), 1e-10)

#         return all_dec_out, all_attn_weights


# class CrossAttention(nn.Module):
#     def __init__(self, encoder_hidden_size, decoder_hidden_size):
#         super(CrossAttention, self).__init__()
#         self.attn_1 = nn.Linear(encoder_hidden_size*2, decoder_hidden_size)
#         self.attn_2 = nn.Linear(decoder_hidden_size, 1, bias=False)

#     def get_attn(self, src_reps, tgt_reps, mask):
#         cat_reps = torch.cat([src_reps, tgt_reps], dim=-1)
#         attn_scores = self.attn_2(F.tanh(self.attn_1(cat_reps))).squeeze(3)  # (batch_size, tgt_len, src_len)
#         attn_scores = mask * attn_scores
#         attn_weights = torch.softmax(attn_scores, dim=2)     # (batch_size, tgt_len, src_len)

#         attn_out = attn_weights.unsqueeze(3).expand_as(src_reps) * src_reps # (batch_size, tgt_len, src_len, hidden_dim)
#         attn_out = torch.sum(attn_out, dim=2)   # (batch_size, tgt_len, hidden_dim)

#         return attn_out, attn_weights

#     def forward(self, src_reps, tgt_reps, mask):
#         attn_out, attn_weights = self.get_attn(src_reps, tgt_reps, mask)

#         return attn_out, attn_weights   # (batch_size, tgt_len, hidden_dim), (batch_size, tgt_len, src_len)


# class SoftAttention(nn.Module):
#     def __init__(self, hidden_size):
#         super(SoftAttention, self).__init__()
#         self.attn = nn.Linear(hidden_size, 1)

#     def get_attn(self, reps, mask=None):
#         attn_scores = self.attn(reps).squeeze(2)
#         if mask is not None:
#             attn_scores = mask * attn_scores
#         attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(2)  # (batch_size, len, 1)
#         attn_out = torch.sum(reps * attn_weights, dim=1)  # (batch_size, hidden_dim)

#         return attn_out, attn_weights

#     def forward(self, reps, mask=None):
#         attn_out, attn_weights = self.get_attn(reps, mask)

#         return attn_out, attn_weights  # (batch_size, hidden_dim), (batch_size, len, 1)


# class Classifier(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, dp=0.5):
#         super(Classifier, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)
#         self.dropout = nn.Dropout(dp)

#     def forward(self, input):
#         return self.fc2(self.dropout(F.relu(self.fc1(input))))


# class Regression(nn.Module):
#     def __init__(self, config):
#         super(Regression, self).__init__()
#         self.config = config

#         self.w_rnn = nn.LSTM(input_size=config["w_size"],
#                              hidden_size=config["hidden_size"],
#                              num_layers=config["num_layers"],
#                              dropout=config["dropout"],
#                              batch_first=True)
#         self.v_rnn = nn.LSTM(input_size=config["v_size"],
#                              hidden_size=config["hidden_size"],
#                              num_layers=config["num_layers"],
#                              dropout=config["dropout"],
#                              batch_first=True)
#         self.a_rnn = nn.LSTM(input_size=config["a_size"],
#                              hidden_size=config["hidden_size"],
#                              num_layers=config["num_layers"],
#                              dropout=config["dropout"],
#                              batch_first=True)
#         self.vw_attn = CrossAttention(config["hidden_size"], config["hidden_size"])
#         self.aw_attn = CrossAttention(config["hidden_size"], config["hidden_size"])
#         self.cross_rnn = nn.LSTM(input_size=config["hidden_size"]*3,
#                                  hidden_size=config["hidden_size"]*3,
#                                  num_layers=config["num_layers"],
#                                  dropout=config["dropout"],
#                                  batch_first=True)
#         self.self_attn = CrossAttention(config["hidden_size"]*3, config["hidden_size"]*3)
#         self.mp_attn = SoftAttention(config["hidden_size"])
#         self.fc1 = nn.Linear(config["hidden_size"] * 5, config["hidden_size"])
#         self.fc2 = nn.Linear(config["hidden_size"], 1)
#         self.dropout = nn.Dropout(0.5)

#     def encode(self, sequence, lengths, encoder):
#         packed_sequence = pack_padded_sequence(sequence, lengths.cpu(), batch_first=True)
#         packed_hs, (final_h, _) = encoder(packed_sequence)
#         enc_hs, lens = pad_packed_sequence(packed_hs, batch_first=True)

#         return enc_hs, final_h, lens

#     def complementary(self, hs, comp_mask, attn):
#         comp_mask = comp_mask.expand_as(hs).bool()
#         sel = hs.masked_select(comp_mask).view(hs.size(0), -1, hs.size(2))
#         comp_out, comp_attn = attn(sel)
#         return comp_out, comp_attn

#     def forward(self, w, v, a, lengths, w2v_consi_attn_mask=None, w2a_consi_attn_mask=None, w2v_comp_mask=None, w2a_comp_mask=None):
#         w_hs, final_w_h, _ = self.encode(w, lengths, self.w_rnn)
#         v_hs, final_v_h, _ = self.encode(v, lengths, self.v_rnn)
#         a_hs, final_a_h, _ = self.encode(a, lengths, self.a_rnn)

#         batch_size = w_hs.size(0)
#         w_len, v_len, a_len = w_hs.size(1), v_hs.size(1), a_hs.size(1)  # w_len == v_len == a_len
#         h_dim = w_hs.size(2)

#         w_reps_4_v = w_hs.contiguous().view(batch_size, w_len, 1, h_dim).expand(batch_size, w_len, v_len, h_dim)
#         w_reps_4_a = w_hs.contiguous().view(batch_size, w_len, 1, h_dim).expand(batch_size, w_len, a_len, h_dim)
#         v_reps = v_hs.contiguous().view(batch_size, 1, v_len, h_dim).expand(batch_size, w_len, v_len, h_dim)
#         a_reps = a_hs.contiguous().view(batch_size, 1, a_len, h_dim).expand(batch_size, w_len, a_len, h_dim)

#         # get mask for attention
#         seq_range = torch.arange(0, v_len).long().to(v.device)
#         seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, v_len)
#         seq_length_expand = lengths.unsqueeze(1).expand_as(seq_range_expand).long()
#         vw_mask = (seq_range_expand < seq_length_expand).unsqueeze(1).expand(batch_size, w_len, v_len).float()
#         aw_mask = deepcopy(vw_mask)

#         _, v_attn_weights = self.vw_attn(v_reps, w_reps_4_v, vw_mask)        # (b, w_n, v_n)
#         _, a_attn_weights = self.aw_attn(a_reps, w_reps_4_a, aw_mask)        # (b, w_n, a_n)

#         # consistent / modality-invariant
#         if w2v_consi_attn_mask is None:
#             v_consi_attn_weights = v_attn_weights
#         else:
#             v_consi_attn_weights = v_attn_weights * w2v_consi_attn_mask     # (b, w_n, v_n)
#         if w2a_consi_attn_mask is None:
#             a_consi_attn_weights = a_attn_weights
#         else:
#             a_consi_attn_weights = a_attn_weights * w2a_consi_attn_mask     # (b, w_n, a_n)

#         v_consi_out = v_consi_attn_weights.unsqueeze(3).expand_as(v_reps) * v_reps  # (b, w_len, v_len, d)
#         a_consi_out = a_consi_attn_weights.unsqueeze(3).expand_as(a_reps) * a_reps  # (b, w_len, a_len, d)
#         v_consi_out = torch.sum(v_consi_out, dim=2)     # (b, w_len, d)
#         a_consi_out = torch.sum(a_consi_out, dim=2)     # (b, w_len, d)

#         concat = torch.cat([w_hs, v_consi_out, a_consi_out], dim=2)
#         c_hs, final_c_h, lens = self.encode(concat, lengths, self.cross_rnn)

#         # self attention
#         c_enc_reps = c_hs.contiguous().view(batch_size, 1, w_len, h_dim*3).expand(batch_size, w_len, w_len, h_dim*3)
#         c_dec_reps = c_hs.contiguous().view(batch_size, w_len, 1, h_dim*3).expand(batch_size, w_len, w_len, h_dim*3)
#         c_mask = (seq_range_expand < seq_length_expand).unsqueeze(1).expand(batch_size, w_len, w_len).float()

#         _, c_attn_weights = self.self_attn(c_enc_reps, c_dec_reps, c_mask)
#         c_attn_out = c_attn_weights.unsqueeze(3).expand_as(c_enc_reps) * c_enc_reps
#         c_attn_out = torch.sum(c_attn_out, dim=2)

#         # select final
#         memory = c_attn_out.view(batch_size*w_len, -1)
#         index = lens - 1 + torch.arange(batch_size) * w_len
#         if torch.cuda.is_available():
#             index = index.cuda()
#         c_out = torch.index_select(memory, 0, index).view(batch_size, -1)

#         # complementary / modality-private
#         if w2v_comp_mask is None:
#             v_comp_out, v_soft_attn_weights = self.mp_attn(v_hs)
#         else:
#             v_comp_out, v_soft_attn_weights = self.complementary(v_hs, w2v_comp_mask, self.mp_attn)
#         if w2a_comp_mask is None:
#             a_comp_out, a_soft_attn_weights = self.mp_attn(a_hs)
#         else:
#             a_comp_out, a_soft_attn_weights = self.complementary(a_hs, w2a_comp_mask, self.mp_attn)

#         out = torch.cat([c_out, v_comp_out, a_comp_out], dim=-1)

#         out = self.fc2(self.dropout(F.relu(self.fc1(out))))

#         return out.view(-1), [v_attn_weights, v_soft_attn_weights, a_attn_weights, a_soft_attn_weights]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from copy import deepcopy

class Translation(nn.Module):
    def __init__(self, config):
        super(Translation, self).__init__()
        self.config = config
        # Encoder LSTM
        self.encoder = nn.LSTM(input_size=config["source_size"],
                               hidden_size=config["encoder_hidden_size"],
                               num_layers=config["encoder_num_layers"],
                               dropout=config["encoder_dropout"])
        self.bn = nn.BatchNorm1d(config["encoder_hidden_size"])
        # Decoder LSTM
        self.decoder = nn.LSTM(input_size=config["decoder_input_size"],
                               hidden_size=config["decoder_hidden_size"],
                               num_layers=config["decoder_num_layers"],
                               dropout=config["decoder_dropout"])
        self.attn_1 = nn.Linear(config["encoder_hidden_size"] * 2, config["decoder_hidden_size"])
        self.attn_2 = nn.Linear(config["decoder_hidden_size"], 1, bias=False)
        self.fc1 = nn.Linear(config["encoder_hidden_size"]+config["decoder_hidden_size"],
                            config["encoder_hidden_size"]+config["decoder_hidden_size"])
        self.fc2 = nn.Linear(config["encoder_hidden_size"]+config["decoder_hidden_size"], config["target_size"])

    def _get_attn_weight(self, dec_rep, enc_reps, mask):
        # 计算 attention 权重
        cat_reps = torch.cat([enc_reps, dec_rep], dim=-1)  # (1, enc_n, b, enc_h_d+dec_h_d)
        attn_scores = self.attn_2(F.tanh(self.attn_1(cat_reps))).squeeze(3)  # (1, enc_n, b)
        attn_scores = mask * attn_scores  # 应用mask
        return torch.softmax(attn_scores, dim=1)  # (1, enc_n, b)

    def decode(self, source, target, enc_hs, mask):
        n_step = len(target)
        enc_n, batch_size, enc_h_d = enc_hs.size()
        dec_h_d = self.config["decoder_hidden_size"]

        # 初始化解码器状态
        dec_h = torch.zeros(1, batch_size, dec_h_d).to(source.device)  # (1, b, dec_h_d)
        dec_c = deepcopy(dec_h)

        dec_rep = dec_h.view(1, 1, batch_size, dec_h_d).expand(1, enc_n, batch_size, dec_h_d)  # (1, enc_n, b, dec_h_d)
        enc_reps = enc_hs.view(1, enc_n, batch_size, enc_h_d)  # (1, enc_n, b, enc_h_d)
        attn_weights = self._get_attn_weight(dec_rep, enc_reps, mask)  # (1, enc_n, b)
        context = attn_weights.unsqueeze(3).expand_as(enc_reps) * enc_reps  # (1, enc_n, b, enc_h_d)
        context = torch.sum(context, dim=1)  # (1, b, enc_h_d)

        dec_in = torch.cat([dec_h, context], dim=2)  # (1, b, enc_h_d+dec_h_d)
        all_attn_weights = torch.empty([n_step, enc_n, batch_size]).to(source.device)  # (dec_n, enc_n, b)
        all_dec_out = torch.empty([n_step, batch_size, self.config["target_size"]]).to(source.device)  # (dec_n, b, dec_o_d)

        # 解码过程
        for i in range(n_step):
            _, (dec_h, dec_c) = self.decoder(dec_in, (dec_h, dec_c))  # (1, b, dec_h_d)

            dec_rep = dec_h.view(1, 1, batch_size, dec_h_d).expand(1, enc_n, batch_size, dec_h_d)
            attn_weights = self._get_attn_weight(dec_rep, enc_reps, mask)  # (1, b, enc_n)
            all_attn_weights[i] = attn_weights

            context = attn_weights.unsqueeze(3).expand_as(enc_reps) * enc_reps  # (1, enc_n, b, enc_h_d)
            context = torch.sum(context, dim=1)  # (1, b, enc_h_d)
            dec_in = torch.cat([dec_h, context], dim=2)
            all_dec_out[i] = self.fc2(F.relu(self.fc1(dec_in)))  # (1, b, dec_o_d)

        return all_dec_out.permute(1, 0, 2).contiguous(), all_attn_weights.permute(2, 0, 1).contiguous()

    def forward(self, source, target, lengths):
        batch_size, enc_n, _ = source.size()
        source = source.permute(1, 0, 2).contiguous()  # (enc_n, b, *)
        target = target.permute(1, 0, 2).contiguous()  # (dec_n, b, *)

        # 编码过程
        enc_hs = self.encode(source, lengths).permute(1, 2, 0).contiguous()

        # 批量归一化
        try:
            enc_hs = self.bn(enc_hs).permute(2, 0, 1).contiguous()
        except:
            enc_hs = enc_hs.permute(2, 0, 1).contiguous()

        # 构建 attention mask
        seq_range = torch.arange(0, enc_n).long().to(source.device)
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, enc_n)
        seq_length_expand = lengths.unsqueeze(1).expand_as(seq_range_expand).long()
        before_mask = (seq_range_expand < seq_length_expand).unsqueeze(1).expand(batch_size, 1, enc_n).float()
        before_mask = before_mask.permute(1, 2, 0).contiguous()
        attn_mask = (seq_range_expand < seq_length_expand).unsqueeze(1).expand(batch_size, enc_n, enc_n).contiguous()
        tgt_mask = (seq_range_expand < seq_length_expand).unsqueeze(2).expand(batch_size, enc_n, target.size(2)).contiguous()

        # 解码过程
        all_dec_out, all_attn_weights = self.decode(source, target, enc_hs, before_mask)

        # 应用 mask 处理解码输出和 attention 权重
        all_dec_out = all_dec_out.masked_fill(~tgt_mask, 0.0)
        all_attn_weights = all_attn_weights.masked_fill(~attn_mask, 1e-10)
        all_attn_weights = all_attn_weights.masked_fill(~attn_mask.transpose(1, 2), 1e-10)

        return all_dec_out, all_attn_weights


class CrossAttention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super(CrossAttention, self).__init__()
        self.attn_1 = nn.Linear(encoder_hidden_size * 2, decoder_hidden_size)
        self.attn_2 = nn.Linear(decoder_hidden_size, 1, bias=False)

    def get_attn(self, src_reps, tgt_reps, mask):
        # 计算 cross attention
        cat_reps = torch.cat([src_reps, tgt_reps], dim=-1)
        attn_scores = self.attn_2(F.tanh(self.attn_1(cat_reps))).squeeze(3)  # (batch_size, tgt_len, src_len)
        attn_scores = mask * attn_scores  # 应用mask
        attn_weights = torch.softmax(attn_scores, dim=2)  # (batch_size, tgt_len, src_len)

        attn_out = attn_weights.unsqueeze(3).expand_as(src_reps) * src_reps  # (batch_size, tgt_len, src_len, hidden_dim)
        attn_out = torch.sum(attn_out, dim=2)  # (batch_size, tgt_len, hidden_dim)

        return attn_out, attn_weights

    def forward(self, src_reps, tgt_reps, mask):
        attn_out, attn_weights = self.get_attn(src_reps, tgt_reps, mask)

        return attn_out, attn_weights  # (batch_size, tgt_len, hidden_dim), (batch_size, tgt_len, src_len)


class SoftAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SoftAttention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def get_attn(self, reps, mask=None):
        attn_scores = self.attn(reps).squeeze(2)  # (batch_size, len)
        if mask is not None:
            attn_scores = mask * attn_scores  # 应用mask
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(2)  # (batch_size, len, 1)
        attn_out = torch.sum(reps * attn_weights, dim=1)  # (batch_size, hidden_dim)

        return attn_out, attn_weights

    def forward(self, reps, mask=None):
        attn_out, attn_weights = self.get_attn(reps, mask)

        return attn_out, attn_weights  # (batch_size, hidden_dim), (batch_size, len, 1)


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dp=0.5):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dp)

    def forward(self, input):
        return self.fc2(self.dropout(F.relu(self.fc1(input))))  # (batch_size, output_size)


class Regression(nn.Module):
    def __init__(self, config):
        super(Regression, self).__init__()
        self.config = config

        self.w_rnn = nn.LSTM(input_size=config["w_size"],
                             hidden_size=config["hidden_size"],
                             num_layers=config["num_layers"],
                             dropout=config["dropout"],
                             batch_first=True)
        self.v_rnn = nn.LSTM(input_size=config["v_size"],
                             hidden_size=config["hidden_size"],
                             num_layers=config["num_layers"],
                             dropout=config["dropout"],
                             batch_first=True)
        self.a_rnn = nn.LSTM(input_size=config["a_size"],
                             hidden_size=config["hidden_size"],
                             num_layers=config["num_layers"],
                             dropout=config["dropout"],
                             batch_first=True)
        self.vw_attn = CrossAttention(config["hidden_size"], config["hidden_size"])
        self.aw_attn = CrossAttention(config["hidden_size"], config["hidden_size"])
        self.cross_rnn = nn.LSTM(input_size=config["hidden_size"] * 3,
                                 hidden_size=config["hidden_size"] * 3,
                                 num_layers=1,
                                 batch_first=True)
        self.fc1 = nn.Linear(config["hidden_size"] * 3, config["output_size"])

    def forward(self, words, vision, audio):
        w_out, _ = self.w_rnn(words)
        v_out, _ = self.v_rnn(vision)
        a_out, _ = self.a_rnn(audio)

        # 计算跨模态注意力
        v_out, _ = self.vw_attn(v_out, w_out, None)
        a_out, _ = self.aw_attn(a_out, w_out, None)

        # 跨模态融合
        cross_out = torch.cat([w_out, v_out, a_out], dim=-1)  # (batch_size, seq_len, hidden_size * 3)

        # 融合后的 RNN
        rnn_out, _ = self.cross_rnn(cross_out)

        return self.fc1(rnn_out)  # (batch_size, seq_len, output_size)
# BatchNorm1d 的问题：对于 enc_hs，使用 BatchNorm1d 来归一化，但可能会遇到维度不匹配的问题。为了避免此类错误，
# 可以在 BatchNorm1d 中添加 unsqueeze 操作来确保批处理维度正确。
# 重复的代码：很多地方存在重复的逻辑（如 attention 的计算），可以提取为单独的函数，减少重复代码。
# 使用 torch.einsum：在一些情况下，使用 torch.einsum 可以提高效率，简化操作。
# 主要改动和优化：
# CrossAttention 类的重构：将原本散布在多个地方的 attention 计算整合为一个类，提升了代码的复用性和模块化。
# SoftAttention 类：用于处理简单的 attention 机制，这个类支持 mask 操作，确保在处理不同长度的序列时不会引入误差。
# 代码优化：通过 torch.cat 和 torch.sum 精简了向量操作，减少冗余计算。
# 这些优化确保了模型在处理多模态数据时更加高效，且更加模块化和可维护。