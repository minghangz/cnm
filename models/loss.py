import torch

def cal_nll_loss(logit, idx, mask, weights=None):
    eps = 0.1
    acc = (logit.max(dim=-1)[1]==idx).float()
    mean_acc = (acc * mask).sum() / mask.sum()
    
    logit = logit.log_softmax(dim=-1)
    nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
    smooth_loss = -logit.sum(dim=-1)
    nll_loss = (1 - eps) * nll_loss + eps / logit.size(-1) * smooth_loss
    if weights is None:
        nll_loss = nll_loss.masked_fill(mask == 0, 0)
        nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)
    else:
        nll_loss = (nll_loss * weights).sum(dim=-1)

    return nll_loss.contiguous(), mean_acc


def rec_loss(words_logit, words_id, words_mask, hard_neg_words_logit=None, **kwargs):
    bsz = words_logit.size(0)
    
    nll_loss, acc = cal_nll_loss(words_logit, words_id, words_mask)
    final_loss = nll_loss.mean()

    if hard_neg_words_logit is not None:
        neg_nll_loss, neg_acc = cal_nll_loss(hard_neg_words_logit, words_id, words_mask) 
        final_loss = final_loss + neg_nll_loss.mean()
        
    loss_dict = {
        'final_loss': final_loss.item(),
        'nll_loss': nll_loss.mean().item(),
    }
    if hard_neg_words_logit is not None:
        loss_dict.update({
            'neg_nll_loss': neg_nll_loss.mean().item(),
            })

    return final_loss, loss_dict
    

def ivc_loss(words_logit, words_id, words_mask, hard_neg_words_logit=None, easy_neg_words_logit=None, **kwargs):
    bsz = words_logit.size(0)

    nll_loss, acc = cal_nll_loss(words_logit, words_id, words_mask)

    if hard_neg_words_logit is not None:
        hard_neg_nll_loss, hard_neg_acc = cal_nll_loss(hard_neg_words_logit, words_id, words_mask)
        tmp_0 = torch.zeros_like(nll_loss).to(words_logit.device)
        tmp_0.requires_grad = False
        hard_neg_loss = torch.max(nll_loss - hard_neg_nll_loss + kwargs["beta_1"], tmp_0)
        loss = hard_neg_loss.mean()
    else:
        loss = nll_loss.mean()
    
    if easy_neg_words_logit is not None:
        easy_neg_nll_loss, easy_neg_acc = cal_nll_loss(easy_neg_words_logit, words_id, words_mask)
        tmp_0 = torch.zeros_like(nll_loss).to(words_logit.device)
        tmp_0.requires_grad = False
        easy_neg_loss = torch.max(nll_loss - easy_neg_nll_loss + kwargs["beta_2"], tmp_0)
        loss = loss + easy_neg_loss.mean()

    return loss, {
        'ivc_loss': loss.item(),
        'easy_neg_loss':  easy_neg_loss.mean().item() if easy_neg_words_logit is not None else 0.0,
        'hard_neg_loss': hard_neg_loss.mean().item() if hard_neg_words_logit is not None else 0.0,
    }
