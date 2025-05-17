import torch
import torch.nn as nn

# Reconstruction loss functions for motion data modeling
class ReConsLoss(nn.Module):
    """
    Class that serves as a customizable loss function module for comparing predicted motion data with ground truth motion data.
    Supports multiple loss types for motion reconstruction:
        - L1 loss (absolute error)
        - L2 loss (mean squared error)
        - Smooth L1 loss (a modified version of L1 that's less sensitive to outliers)
    During training, the loss is calculated in 2 components:
        - Full motion reconstruction error using `Loss(pred_motion, gt_motion)`
        - Joint-specific error using `Loss.forward_joint(pred_motion, gt_motion)`
    These loss components are combined with a commitment loss (from the VQ-VAE architecture) to form the total loss for optimization and are tracked and logged during training.
    """
    def __init__(self, recons_loss, nb_joints):
        super(ReConsLoss, self).__init__()
        
        if recons_loss == 'l1': 
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2' : 
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth' : 
            self.Loss = torch.nn.SmoothL1Loss()
        
        # 4 global motion associated to root (global motion via root position/orientation)
        # 12 local motion (3 local xyz, 3 vel xyz, 6 rot6d) -> xyz position, velocity and rotation
        # 3 global vel xyz
        # 4 foot contact
        self.nb_joints = nb_joints
        self.motion_dim = (nb_joints - 1) * 12 + 4 + 3 + 4
        
    def forward(self, motion_pred, motion_gt) : 
        loss = self.Loss(motion_pred[..., : self.motion_dim], motion_gt[..., :self.motion_dim])
        return loss
    
    def forward_joint(self, motion_pred, motion_gt) : 
        loss = self.Loss(motion_pred[..., 4 : (self.nb_joints - 1) * 3 + 4], motion_gt[..., 4 : (self.nb_joints - 1) * 3 + 4])
        return loss
    
    