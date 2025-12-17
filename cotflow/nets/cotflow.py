import math
import torch
import pytorch_lightning as pl
import torchvision as tv
import normflows as nf
import wandb

from cotflow.nets.autoencoder import AbsAutoencoder


class CoTFlow(torch.nn.Module):
    def __init__(
        self, 
        num_bases=66,
        symmetry_dim=12,
        latent_dim=12,
        flow_layers=24,
        flow_hidden_dim=192,
        eps_p=1e-3,
        eps_q=1e-1,
        scale_map="exp_clamp",
    ):
        super().__init__()

        self.symmetry_dim = symmetry_dim
        self.latent_dim = latent_dim
        self.num_bases = num_bases

        input_dim = latent_dim
        hidden_dim = flow_hidden_dim
        half_dim = input_dim // 2


        base = nf.distributions.base.DiagGaussian(input_dim, trainable=False)

        flows = []
        for i in range(flow_layers):
            # Neural network with two hidden layers having 64 units each
            # Last layer is initialized by zeros making training more stable
            param_map = nf.nets.MLP([half_dim, hidden_dim, hidden_dim, input_dim], init_zeros=True)
            # Add flow layer
            flows.append(nf.flows.AffineCouplingBlock(param_map, scale_map=scale_map))
            # Swap dimensions
            flows.append(nf.flows.Permute(input_dim, mode='swap'))

        self.flow = nf.NormalizingFlow(base, flows)

        self.W = torch.nn.Parameter(torch.randn(num_bases, symmetry_dim, symmetry_dim) * (1.0 / math.sqrt(symmetry_dim)))

        self.log_var_diag = torch.nn.Parameter(torch.zeros(input_dim))

        self.register_buffer('eps_p', torch.tensor(eps_p))
        self.register_buffer('eps_q', torch.tensor(eps_q))

    def parameters(self, recurse = True):
        yield from self.flow.parameters(recurse)
        yield self.W
        yield self.log_var_diag

    def pullback_tangent(self, z: torch.Tensor, v: torch.Tensor):
        """
        Args:
            z: (batch_size, input_dim)
            v: (batch_size, output_dim, input_dim)
        Returns:
            J: (batch_size, output_dim, input_dim)
        """
        def _forward_flow(z: torch.Tensor):
            z = z.unsqueeze(0)
            y = self.flow.forward(z)
            return y.squeeze(0)
        
        def _forward_single(z: torch.Tensor, v: torch.Tensor):
            # z: (input_dim,)
            # v: (num_bases, input_dim)
            J = torch.func.vmap(torch.func.vjp(_forward_flow, z)[1])(v)  # (num_bases, input_dim)
            return J[0]
        J = torch.func.vmap(_forward_single)(z, v)  # (batch_size, num_bases, input_dim)
        return J

    def kl_divergence(self, J_p: torch.Tensor, J_q: torch.Tensor):
        """
        Args:
            J_p: (batch_size, output_dim, input_dim)
            J_q: (batch_size, num_bases, symmetry_dim)
        Returns:
            kl: (batch_size,)
        """

        input_dim = J_p.size(-1)
        output_dim = J_p.size(-2)

        J_p = J_p / J_p.square().sum().sqrt()

        S_p = torch.einsum('bni,bmi->bnm', J_p, J_p)                            # (batch_size, output_dim, output_dim)
        S_q = torch.einsum('bin,bim->bnm', J_q, J_q)                            # (batch_size, symmetry_dim, symmetry_dim)
        S_pq = torch.einsum('bni,bmi->bnm', J_p[:,:,:self.symmetry_dim], J_q)   # (batch_size, output_dim, num_bases)
        D = self.log_var_diag.neg().exp()                                       # (input_dim,)

        I_p = torch.eye(output_dim, device=J_p.device)                      # (output_dim, output_dim)
        I_q = torch.eye(self.symmetry_dim, device=J_q.device)               # (symmetry_dim, symmetry_dim)
        M = S_p + self.eps_p * I_p                                          # (batch_size, output_dim, output_dim)
        H = S_q + D[:self.symmetry_dim] * I_q                               # (batch_size, symmetry_dim, symmetry_dim)

        norm_M = M.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-6)
        norm_H = H.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-6)
        M = M + 1e-3 * norm_M.unsqueeze(-1).unsqueeze(-1) * I_p
        H = H + 1e-3 * norm_H.unsqueeze(-1).unsqueeze(-1) * I_q
        D = D + 1e-3 * norm_H.unsqueeze(-1)

        trace_p = torch.einsum('bij,bij,bj->b', J_p, J_p, D)                # (batch_size,)
        trace_q = torch.einsum('bij,bij->b', J_q, J_q)                      # (batch_size,)
        trace_pq = torch.einsum('bij,bij->b', S_pq, S_pq)                   # (batch_size,)
        trace = (self.eps_p + 1e-3 * norm_M) * (D.sum(-1) + trace_q) + trace_p + trace_pq       # (batch_size,)

        L_M = torch.linalg.cholesky(M)
        L_H = torch.linalg.cholesky(H)
        logdet_M = 2 * torch.log(torch.diagonal(L_M, dim1=-2, dim2=-1)).sum(-1)
        logdet_H = 2 * torch.log(torch.diagonal(L_H, dim1=-2, dim2=-1)).sum(-1)
        logdet_p = logdet_M + (input_dim - output_dim) * self.eps_p.log()   # (batch_size,)
        logdet_q = logdet_H + D[:, self.symmetry_dim:].log().sum(-1)        # (batch_size,)
        logdet = -(logdet_p + logdet_q)

        kl = 0.5 * (trace + logdet - input_dim)

        return kl

    @torch.no_grad()
    def sample(self, num_samples):
        z = self.flow.sample(num_samples)[0]
        return z
    
    def forward(self, y: torch.Tensor, J_p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_prob = self.flow.log_prob(y)

        z = self.flow.inverse(y)

        J_p = self.pullback_tangent(z, J_p)  # (B, output_dim, input_dim)

        J_q = torch.einsum('bi,mji->bmj', z[:, :self.symmetry_dim], self.W - self.W.mT)  # (B, num_bases, symmetry_dim)

        kl = self.kl_divergence(J_p, J_q)  # (B,)

        return log_prob, kl


class CoTFlowModule(pl.LightningModule):
    def __init__(
        self,
        autoencoder: AbsAutoencoder,
        flow_kwargs,
        sample_num=64,
        **optimizer_kwargs,
    ):
        super().__init__()
        self.autoencoder = autoencoder

        self.model = CoTFlow(**flow_kwargs)
        self.sample_num = sample_num

        self.optimizer_kwargs = optimizer_kwargs
    
    def forward(self, x):
        return self.model(x)
    
    def on_validation_epoch_end(self):
        # 検証終了時に画像生成しwandbに記録
        with torch.no_grad():
            z = self.model.sample(self.sample_num)
            img = self.autoencoder.decode(z)
            img = img.clamp(0, 1)
            grid = tv.utils.make_grid(img.cpu(), nrow=8)
            wandb_logger = self.logger
            if hasattr(wandb_logger, "experiment"):
                wandb_logger.experiment.log({f"image/sample": wandb.Image(grid, caption=f"epoch {self.current_epoch}")})

    def training_step(self, batch, batch_idx):
        z, jac_predictor = batch
        log_prob, kl = self.model.forward(z, jac_predictor)
        log_prob = log_prob.mean()
        kl = kl.mean()
        loss = kl - log_prob
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_log_prob', log_prob, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_kl', kl, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        z, jac_predictor = batch
        log_prob, kl = self.model.forward(z, jac_predictor)
        log_prob = log_prob.mean()
        kl = kl.mean()
        loss = kl - log_prob
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_log_prob', log_prob, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_kl', kl, on_step=False, on_epoch=True, prog_bar=False)
        return loss    

    def configure_optimizers(self):
        optimizer = torch.optim.Adamax(self.model.parameters(), **self.optimizer_kwargs)

        return optimizer
