import torch.nn as nn
import torch


class GeodesicLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, m1, m2):
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1)/2
        theta = torch.acos(torch.clamp(cos, -1+self.eps, 1-self.eps))

        return torch.mean(theta)


import torch
import torch.nn as nn

class GeodesicAndFrobeniusLoss(nn.Module):
    def __init__(self, geodesic_weight=1.0, frobenius_weight=1.0, eps=1e-7):
        super().__init__()
        self.geodesic_weight = geodesic_weight
        self.frobenius_weight = frobenius_weight
        self.eps = eps

    def forward(self, m1, m2):
        # Compute the relative rotation matrix
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

        # Geodesic Loss
        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        theta = torch.acos(torch.clamp(cos, -1 + self.eps, 1 - self.eps))
        geodesic_loss = torch.mean(theta)

        # Frobenius Loss
        eye = torch.eye(3, device=m.device).expand_as(m)
        frobenius_loss = torch.mean(torch.norm(m - eye, dim=(1, 2)))

        # Combine both losses
        combined_loss = (self.geodesic_weight * geodesic_loss +
                         self.frobenius_weight * frobenius_loss)

        return combined_loss
