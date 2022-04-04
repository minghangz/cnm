import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import DualTransformer

class CNM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = config['dropout']
        self.vocab_size = config['vocab_size']
        self.alpha = config["alpha"]
        self.use_negative = config['use_negative']
        self.max_width = config['max_width'] if 'max_width' in config else 1

        self.frame_fc = nn.Linear(config['frames_input_size'], config['hidden_size'])
        self.word_fc = nn.Linear(config['words_input_size'], config['hidden_size'])
        self.mask_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        self.start_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        
        self.trans = DualTransformer(**config['DualTransformer']) # we share the parameters of mask generator and mask conditioned reconstructor 
        self.fc_rec = nn.Linear(config['hidden_size'], self.vocab_size)
        self.fc_mask_gen = nn.Linear(config['hidden_size'], 2)

        self.word_pos_encoder = SinusoidalPositionalEmbedding(config['hidden_size'], 0, config['max_num_words'])

    def froze_mask_generator(self):
        for name, param in self.named_parameters():
            if 'mask_gen' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    def froze_reconstructor(self):
        for name, param in self.named_parameters():
            if 'mask_gen' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def unfroze(self):
        for name, param in self.named_parameters():
            param.requires_grad = True

    def forward(self, frames_feat, frames_len, words_id, words_feat, words_len, weights, **kwargs):
        bsz, n_frames, _ = frames_feat.shape
        frames_feat = F.dropout(frames_feat, self.dropout, self.training)
        frames_feat = self.frame_fc(frames_feat)
        frames_mask = _generate_mask(frames_feat, frames_len)

        words_feat[:, 0] = self.start_vec
        words_pos = self.word_pos_encoder(words_feat)
        words_feat = F.dropout(words_feat, self.dropout, self.training)
        words_feat = self.word_fc(words_feat)
        words_mask = _generate_mask(words_feat, words_len + 1)

        enc_out, h = self.trans(frames_feat, frames_mask, words_feat + words_pos, words_mask, decoding=1)
        gauss_param = torch.sigmoid(self.fc_mask_gen(h[:, -1])).view(bsz, 2)
        gauss_center = gauss_param[:, 0]
        gauss_width = gauss_param[:, 1] * self.max_width
        if not self.training:
            return {
                'width': gauss_width,
                'center': gauss_center,
            } 

        # downsample for effeciency
        props_len = n_frames//4
        keep_idx = torch.linspace(0, n_frames-1, steps=props_len).long()
        props_feat = frames_feat[:, keep_idx]
        props_mask = frames_mask[:, keep_idx]
        
        gauss_weight = self.generate_gauss_weight(props_len, gauss_center, gauss_width)

        # semantic completion
        words_feat1, masked_words = self._mask_words(words_feat, words_len, weights=weights)
        words_feat1 = words_feat1 + words_pos
        words_feat1 = words_feat1[:, :-1]
        words_mask1 = words_mask[:, :-1]

        _, h, attn_weight = self.trans(props_feat, props_mask, words_feat1, words_mask1, decoding=2, gauss_weight=gauss_weight, need_weight=True)
        words_logit = self.fc_rec(h)

        if self.use_negative:
            _, hard_neg_h = self.trans(props_feat, props_mask, words_feat1, words_mask1, decoding=2)
            hard_neg_words_logit = self.fc_rec(hard_neg_h)

            _, easy_neg_h = self.trans(props_feat, props_mask, words_feat1, words_mask1, decoding=2, gauss_weight=1-gauss_weight)
            easy_neg_words_logit = self.fc_rec(easy_neg_h)
        else:
            hard_neg_words_logit = None
            easy_neg_words_logit = None

        weights = None
        return {
            'hard_neg_words_logit': hard_neg_words_logit,
            'easy_neg_words_logit': easy_neg_words_logit,
            'words_logit': words_logit, 
            'words_id': words_id,
            'weights': weights,
            'words_mask': words_mask[:, :-1],
            'width': gauss_width,
            'center': gauss_center,
            'gauss_weight': gauss_weight,
            'attn_weight': attn_weight,
        }
    

    def generate_gauss_weight(self, props_len, center, width):
        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        center = center.unsqueeze(-1)
        width = width.unsqueeze(-1).clamp(0.1)
        weight = torch.exp(-self.alpha*(weight - center).pow(2)/width.pow(2))
        return weight

    def _mask_words(self, words_feat, words_len, weights=None):
        token = self.mask_vec.unsqueeze(0).unsqueeze(0)
        token = self.word_fc(token)

        masked_words = []
        for i, l in enumerate(words_len):
            l = int(l)
            num_masked_words = max(l // 3, 1) 
            masked_words.append(torch.zeros([words_feat.size(1)]).byte().to(words_feat.device))
            p = weights[i, :l].cpu().numpy() if weights is not None else None
            choices = np.random.choice(np.arange(1, l + 1), num_masked_words, replace=False, p=p)
            masked_words[-1][choices] = 1

        masked_words = torch.stack(masked_words, 0).unsqueeze(-1)
        masked_words_vec = words_feat.new_zeros(*words_feat.size()) + token
        masked_words_vec = masked_words_vec.masked_fill_(masked_words == 0, 0)
        words_feat1 = words_feat.masked_fill(masked_words == 1, 0) + masked_words_vec
        return words_feat1, masked_words

def _generate_mask(x, x_len):
    mask = []
    for l in x_len:
        mask.append(torch.zeros([x.size(1)]).byte().to(x_len.device))
        mask[-1][:l] = 1
    mask = torch.stack(mask, 0)
    return mask
    

class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        import math
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, **kwargs):
        bsz, seq_len, _ = input.size()
        max_pos = seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(input.device)[:max_pos]
        return self.weights.unsqueeze(0)

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number
