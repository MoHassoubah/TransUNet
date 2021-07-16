import logging
import torch
from torch import nn
from torch.nn import functional as F

from NCE.NCECriterion import NCECriterion

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())
            
    def forward(self, emb_i, emb_j):
        #input  batch size x hidden_dim
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)#take the diagonal after batch size steps from the diagonal, upper the diagonal
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss
    

class LearnedLoss():
    def __init__(self, losstype, ndata=None, batch_size=None):
 
        
        if losstype == 'CrossEntropy':
            self.lossF = torch.nn.CrossEntropyLoss()
            self.adj = 1
        elif losstype == 'BinaryCrossEntropy':
            self.lossF = torch.nn.BCEWithLogitsLoss()
            self.adj = 1
        elif losstype == 'L1':
            self.lossF = torch.nn.L1Loss()
            self.adj = 0.5
        elif losstype == 'L2':
            self.lossF = torch.nn.MSELoss()
            self.adj = 0.5
        elif losstype == 'Contrastive':
            self.lossF = ContrastiveLoss(batch_size)
            self.adj = 1
        elif losstype == 'NCE':
            self.lossF = NCECriterion(ndata).cuda()
            self.adj = 1


    def calculate_loss(self, output, label_or_index):
        return self.lossF(output, label_or_index)
        

    def calculate_weighted_loss(self, loss, _s):
        w_loss =  (self.adj * torch.exp(-_s) * loss) + (0.5 * _s) #_s is log(std^2)
        # if w_loss.item()>0:
        return w_loss
        # else:
            # return w_loss*0
    
class MTLLOSS():
    def __init__(self, loss_funcs, device):
        super().__init__()
        self._loss_funcs = loss_funcs
        self.device = device

    def __call__(self, output_contrastive, index,
                 output_recons, target_recons,nce_w, reconstruction_w, contrastive_prd, weight_prev_cycle, nce_converge_w):
        """Returns (overall loss, [seperate task losses])"""

        
        # cn_loss = self._loss_funcs[0].calculate_loss(output_contrastive, target_contrastive)        
        # contrastive_loss = self._loss_funcs[0].calculate_weighted_loss(cn_loss, contrastive_w) 
        
        nce_loss = self._loss_funcs[2].calculate_loss(output_contrastive, index)     
        w_nce_loss = self._loss_funcs[2].calculate_weighted_loss(nce_loss, nce_w) 
                
        rec_loss = self._loss_funcs[1].calculate_loss(output_recons, target_recons)
        reconstruction_loss = self._loss_funcs[1].calculate_weighted_loss(rec_loss, reconstruction_w) 
        
        norm_contrastive_prd = F.normalize(contrastive_prd, dim=1)
        # print("norm_contrastive_prd")
        # print(norm_contrastive_prd.shape)
        # print("weight_prev_cycle")
        # print(weight_prev_cycle.shape)
        convergence_loss = self._loss_funcs[3].calculate_loss(norm_contrastive_prd, weight_prev_cycle)
        w_conv=60
        w_convergence_loss = w_conv*convergence_loss#self._loss_funcs[3].calculate_weighted_loss(convergence_loss, nce_converge_w) 
                
        reg_term = torch.pow(contrastive_prd,2).sum()#torch.pow(norm_contrastive_prd, 2)   
        reg_loss=1./reg_term
        
        # total_loss = contrastive_loss + reconstruction_loss
        total_loss = w_nce_loss + reconstruction_loss + w_convergence_loss + (w_conv*reg_loss)
        
        logging.info('w_nce_loss : %f,******** w_convergence_loss : %f, ******** reconstruction_loss : %f' % (w_nce_loss.item(), \
                w_convergence_loss.item(), reconstruction_loss.item()))
        return total_loss, (nce_loss, convergence_loss, rec_loss)

    
def MTL_loss(device, ndata, batch_size):
    """Returns the learned uncertainty loss function."""

    task_contrastive  = LearnedLoss('Contrastive', batch_size=batch_size) 
    task_recons = LearnedLoss('L1') 
    task_NCE = LearnedLoss('NCE', ndata=ndata) 
    task_nce_converge = LearnedLoss('L2') 
    return MTLLOSS([task_contrastive, task_recons, task_NCE,task_nce_converge], device)

        