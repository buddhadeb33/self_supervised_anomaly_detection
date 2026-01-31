import torch
import torch.nn.functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    Normalized temperature-scaled cross entropy loss for SimCLR.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)

    sim = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim = sim.masked_fill(mask, float("-inf"))

    labels = (torch.arange(2 * batch_size, device=z.device) + batch_size) % (
        2 * batch_size
    )
    loss = F.cross_entropy(sim, labels)
    return loss

