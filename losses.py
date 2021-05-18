import torch
from torch import nn
from torch.nn import functional as F

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
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)#take the diagonal after batch size steps from the diagonal upper the diagonal
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss
    

class LearnedLoss():
    def __init__(self, losstype, batch_size=None):
 
        
        if losstype == 'CrossEntropy':
            self.lossF = torch.nn.CrossEntropyLoss()
            self.adj = 1
        elif losstype == 'BinaryCrossEntropy':
            self.lossF = torch.nn.BCEWithLogitsLoss()
            self.adj = 1
        elif losstype == 'L1':
            self.lossF = torch.nn.L1Loss()
            self.adj = 0.5
        elif losstype == 'Contrastive':
            self.lossF = ContrastiveLoss(batch_size)
            self.adj = 1


    def calculate_loss(self, output, label):
        return self.lossF(output, label)

    def calculate_weighted_loss(self, loss, _s):
        w_loss =  (self.adj * torch.exp(-_s) * loss) + (0.5 * _s) #_s is log(std^2)
        if w_loss.item()>0:
            return w_loss
        else:
            return w_loss*0
    
class MTLLOSS():
    def __init__(self, loss_funcs, device):
        super().__init__()
        self._loss_funcs = loss_funcs
        self.device = device

    def __call__(self, output_rot, target_rot,output_rot_axis, target_rot_axis, output_contrastive, target_contrastive, 
                 output_recons, target_recons, rot_w, rot_axis_w,contrastive_w, reconstruction_w):
        """Returns (overall loss, [seperate task losses])"""

        
        r_loss = self._loss_funcs[0].calculate_loss(output_rot, target_rot)
        rotation_loss = self._loss_funcs[0].calculate_weighted_loss(r_loss, rot_w) 
        
        if len(target_rot_axis)>0:
            r_axis_loss = self._loss_funcs[3].calculate_loss(output_rot_axis, target_rot_axis)
            rotation_axis_loss = self._loss_funcs[3].calculate_weighted_loss(r_axis_loss, rot_axis_w) 
        else:
            rotation_axis_loss=r_axis_loss = torch.tensor(0)
            
        cn_loss = self._loss_funcs[1].calculate_loss(output_contrastive, target_contrastive)        
        contrastive_loss = self._loss_funcs[1].calculate_weighted_loss(cn_loss, contrastive_w) 
                
        rec_loss = self._loss_funcs[2].calculate_loss(output_recons, target_recons)
        reconstruction_loss = self._loss_funcs[2].calculate_weighted_loss(rec_loss, reconstruction_w) 
                
        total_loss = rotation_loss + rotation_axis_loss + contrastive_loss + reconstruction_loss

        return total_loss, (r_loss, r_axis_loss, cn_loss, rec_loss)

    
def MTL_loss(device, batch_size):
    """Returns the learned uncertainty loss function."""

    task_rot = LearnedLoss('BinaryCrossEntropy') 
    task_rot_axis = LearnedLoss('BinaryCrossEntropy') 
    task_contrastive  = LearnedLoss('Contrastive', batch_size) 
    task_recons = LearnedLoss('L1') 
    return MTLLOSS([task_rot, task_contrastive, task_recons,task_rot_axis], device)

        