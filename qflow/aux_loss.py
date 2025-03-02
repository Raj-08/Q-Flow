import torch
import torch.nn.functional as F

class AuxiliaryLosses:
    @staticmethod
    def kl_divergence_loss(logits_old, logits_new, mask):
        """
        Computes KL divergence between old and new policy distributions.
        :param logits_old: Logits from the old policy
        :param logits_new: Logits from the current policy
        :param mask: Attention mask to exclude padding tokens
        """
        probs_old = F.log_softmax(logits_old, dim=-1).exp()
        log_probs_new = F.log_softmax(logits_new, dim=-1)
        
        kl_div = (probs_old * (F.log_softmax(logits_old, dim=-1) - log_probs_new)).sum(dim=-1)
        kl_div = (kl_div * mask).sum() / mask.sum()
        return kl_div

    @staticmethod
    def entropy_regularizer(logits, mask):
        """
        Computes entropy regularization term to encourage exploration.
        :param logits: Logits from the policy model
        :param mask: Attention mask to exclude padding tokens
        """
        probs = F.log_softmax(logits, dim=-1).exp()
        entropy = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        entropy = (entropy * mask).sum() / mask.sum()
        return entropy
