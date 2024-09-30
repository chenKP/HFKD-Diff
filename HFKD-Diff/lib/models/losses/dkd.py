import torch
import torch.nn.functional as F
from torch import nn
class DKD(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, tau=1.0):
        super(DKD, self).__init__()
        self.beta = beta
        self.alpha = alpha
        self.tau = tau

    def forward(self, logits_student, logits_teacher, target):
        temperature = self.tau
        
        if logits_student.dim() == 4:
            b, c, _,_ = logits_student.shape
            logits_student = logits_student.view(b, c)
        if logits_teacher.dim() == 4:
            b, c, _,_ = logits_teacher.shape
            logits_teacher = logits_teacher.view(b, c)
            
        gt_mask = self._get_gt_mask(logits_student, target)
        other_mask = self._get_other_mask(logits_student, target)
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        pred_student = self.cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = self.cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        tckd_loss = (
                F.kl_div(log_pred_student, pred_teacher, reduction='batchmean')
                * (temperature ** 2)
        )
        pred_teacher_part2 = F.softmax(
            logits_teacher / temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / temperature - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
                F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='batchmean')
                * (temperature ** 2)
        )
        return self.alpha * tckd_loss + self.beta * nckd_loss


    def _get_gt_mask(self,logits, target):
        target = target.reshape(-1)
        mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
        return mask


    def _get_other_mask(self, logits, target):
        target = target.reshape(-1)
        mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
        return mask


    def cat_mask(self, t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        rt = torch.cat([t1, t2], dim=1)
        return rt
