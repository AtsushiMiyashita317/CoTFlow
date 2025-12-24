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


class CoTGlow(torch.nn.Module):
    def __init__(
        self, 
        pretrained_model: torch.nn.Module,
        input_shape,
        output_shape,
        num_bases=64,
        hidden_channels=512,
        n_levels=6,
        n_blocks=32,
        eps=1e-3,
        scale_map="exp_clamp",
        x_only=False,
    ):
        super().__init__()

        self.pretrained_model = pretrained_model.eval()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.x_only = x_only

        self.num_bases = num_bases
        self.n_levels = n_levels

        L = n_levels
        K = n_blocks
        channels = input_shape[0]

        q0 = []
        merges = []
        flows = []
        self.latent_shapes = []
        for i in range(L):
            flows_ = []
            for j in range(K):
                flows_ += [nf.flows.GlowBlock(
                    channels * 2 ** (L + 1 - i), 
                    hidden_channels=hidden_channels,
                    split_mode='channel', 
                    scale=True,
                    scale_map=scale_map
                )]
            flows_ += [nf.flows.Squeeze()]
            flows += [flows_]
            if i > 0:
                merges += [nf.flows.Merge()]
                latent_shape = (input_shape[0] * 2 ** (L - i), input_shape[1] // 2 ** (L - i), 
                                input_shape[2] // 2 ** (L - i))
            else:
                latent_shape = (input_shape[0] * 2 ** (L + 1), input_shape[1] // 2 ** L, 
                                input_shape[2] // 2 ** L)
            self.latent_shapes.append(latent_shape)
            q0 += [nf.distributions.DiagGaussian(latent_shape, trainable=False)]

        transform = nf.transforms.Logit()

        # Construct flow model with the multiscale architecture
        self.flow = nf.MultiscaleFlow(q0, flows, merges, transform)

        latent_channels = [shape[0] for shape in self.latent_shapes]
        latent_channels_total = sum(latent_channels)

        L_data = torch.randn(num_bases, latent_channels_total, latent_channels_total) * (1.0 / math.sqrt(latent_channels_total))
        self.L = torch.nn.Parameter(L_data)

        lam_data = torch.randn(num_bases, input_shape[1] // 2, input_shape[2] // 2)
        self.lam = torch.nn.Parameter(lam_data)

        self.register_buffer('eps', torch.tensor(eps))

    def parameters(self, recurse = True):
        yield from self.flow.parameters(recurse)
        yield self.L
        yield self.lam

    def forward_model(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0)
        y = self.pretrained_model(x)
        return y.squeeze(0)

    def forward_flow(self, *z: torch.Tensor) -> torch.Tensor:
        z = [zi.unsqueeze(0) for zi in z]
        x, _ = self.flow.forward_and_log_det(z)
        return x.squeeze(0)

    def inverse_flow(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = x.unsqueeze(0)
        z, _ = self.flow.inverse_and_log_det(x)
        return [zi.squeeze(0) for zi in z]

    def forward_fft(self, *z: torch.Tensor) -> torch.Tensor:
        z_upsampled = []
        for zi in z:
            hi, wi = zi.size(-2), zi.size(-1)
            zi_upsampled = torch.zeros((zi.size(0), zi.size(1), z[-1].size(-2), z[-1].size(-1)), device=zi.device)
            rh = z[-1].size(-2) // hi
            rw = z[-1].size(-1) // wi
            zi_upsampled[..., ::rh, ::rw] = zi
            z_upsampled.append(zi_upsampled)
        z_upsampled = torch.cat(z_upsampled, dim=-3)  # (batch_size, channels_total, H, W)
        z_fft = torch.fft.fftn(z_upsampled, dim=(-2, -1))  # (batch_size, channels_total, H, W)
        return z_fft

    def inverse_fft(self, z_fft: torch.Tensor) -> list[torch.Tensor]:
        z = torch.fft.ifftn(z_fft, dim=(-2, -1)).real  # (batch_size, channels_total, H, W)
        
        latent_channels = [shape[0] for shape in self.latent_shapes]
        z_list = torch.split(z, latent_channels, dim=-3)

        h_target = z_fft.size(-2)
        w_target = z_fft.size(-1)
        z = []
        for zi, shape_i in zip(z_list, self.latent_shapes):
            hi, wi = shape_i[1], shape_i[2]
            rh = h_target // hi
            rw = w_target // wi
            z.append(zi[..., ::rh, ::rw])
        return z
    
    def forward_vector_field(self, *z: torch.Tensor) -> list[torch.Tensor]:
        z_fft = self.forward_fft(*z)
        L = self.L - self.L.mT
        L = L.cfloat()
        lam = self.lam - self.lam.flip([1, 2]).roll(shifts=[1, 1], dims=[1, 2])
        lam = lam * 1j
        v_fft = torch.einsum('mcd,bchw->bmdhw', L, z_fft)  # (batch_size, num_bases, channels_total, H, W)
        v_fft = torch.einsum('mhw,bmchw->bmchw', lam, v_fft)  # (batch_size, num_bases, channels_total, H, W)
        v = self.inverse_fft(v_fft)  # list of (batch_size, num_bases, channels_i, H_i, W_i)
        return v
    
    def sample_cotangent(self, x: torch.Tensor) -> torch.Tensor:
        def _sample_cotangent_single(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            _, func = torch.func.vjp(self.forward_model, x)
            return func(w)[0]
        
        batch_size = x.size(0)
        w = torch.randn(batch_size, *self.output_shape, device=x.device)
        w = torch.func.vmap(_sample_cotangent_single)(x, w)
        return w

    def pullback_cotangent(self, w: torch.Tensor, z: list[torch.Tensor]) -> list[torch.Tensor]:
        def _pullback_cotangent_single(w: torch.Tensor, *z: torch.Tensor) -> torch.Tensor:
            _, vjp_fn = torch.func.vjp(self.forward_flow, *z)

            w = vjp_fn(w)

            return [wi.squeeze(0) for wi in w]

        w = torch.func.vmap(_pullback_cotangent_single)(w, *z)

        return w

    def log_prob(self, w: list[torch.Tensor], z: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            wi: (batch_size, channels_i, height_i, width_i)
            zi: (batch_size, channels_i, height_i, width_i)
        Returns:
            log_prob: (batch_size,)
        """
        device = z[0].device
        batch_size = z[0].size(0)
        num_bases = self.num_bases

        S_ww = torch.zeros(batch_size, device=device)
        for wi in w:
            S_ww = S_ww + torch.einsum('bchw,bchw->b', wi, wi)

        v = self.forward_vector_field(*z)  # list of (batch_size, num_bases, channels_i, H_i, W_i)

        S_zz = torch.zeros(batch_size, num_bases, num_bases, device=device)
        for vi in v:
            S_zz = S_zz + torch.einsum('bnchw,bmchw->bnm', vi, vi)
        
        S_zw = torch.zeros(batch_size, num_bases, device=device)
        for vi, wi in zip(v, w):
            S_zw = S_zw + torch.einsum('bnchw,bchw->bn', vi, wi)

        var = S_zz.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-6)      # (batch_size,)
        std = var.sqrt()
        S_zz = S_zz / var.unsqueeze(-1).unsqueeze(-1)
        S_zw = S_zw / std.unsqueeze(-1)

        I = torch.eye(num_bases, device=device)
        M = S_zz + self.eps.unsqueeze(-1).unsqueeze(-1) * I               # (batch_size, num_bases, num_bases)

        # w^t(eI + v^tv)w= e w^tw + w^tv^tvw
        trace = self.eps * S_ww + torch.einsum('bn,bn->b', S_zw, S_zw)    # (batch_size,)

        L_H = torch.linalg.cholesky(M)
        logdet = 2 * torch.log(torch.diagonal(L_H, dim1=-2, dim2=-1)).sum(-1)

        log_prob = 0.5 * (logdet - trace)

        return log_prob

    @torch.no_grad()
    def sample(self, num_samples):
        x = self.flow.sample(num_samples)[0]
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w = self.sample_cotangent(x).detach()  # (B, output_dim, input_dim)

        z, logdet = self.flow.inverse_and_log_det(x)
        log_prob_x = logdet  # (B,)
        for zi, q0i in zip(z, self.flow.q0):
            log_prob_x = log_prob_x + q0i.log_prob(zi)

        if self.x_only:
            return log_prob_x, torch.zeros_like(log_prob_x)

        w = self.pullback_cotangent(w, z)

        log_prob_w = self.log_prob(w, z) - logdet  # (B,)

        return log_prob_x, log_prob_w


class CoTGlowModule(pl.LightningModule):
    def __init__(
        self,
        pretrained_model: torch.nn.Module,
        flow_kwargs,
        sample_num=64,
        **optimizer_kwargs,
    ):
        super().__init__()

        self.model = CoTGlow(pretrained_model, **flow_kwargs)
        self.sample_num = sample_num

        self.optimizer_kwargs = optimizer_kwargs
    
    def forward(self, x):
        return self.model(x)
    
    def on_validation_epoch_end(self):
        # 検証終了時に画像生成しwandbに記録
        with torch.no_grad():
            img = self.model.sample(self.sample_num)
            img = img.clamp(0, 1)
            grid = tv.utils.make_grid(img.cpu(), nrow=8)
            wandb_logger = self.logger
            if hasattr(wandb_logger, "experiment"):
                wandb_logger.experiment.log({f"image/sample": wandb.Image(grid, caption=f"epoch {self.current_epoch}")})

    def training_step(self, batch, batch_idx):
        x = batch['image']
        log_prob_x, log_prob_w = self.model.forward(x)
        log_prob_x = log_prob_x.mean()
        log_prob_w = log_prob_w.mean()
        loss = - (log_prob_x + log_prob_w)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_log_prob_x', log_prob_x, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_log_prob_w', log_prob_w, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        log_prob_x, log_prob_w = self.model.forward(x)
        log_prob_x = log_prob_x.mean()
        log_prob_w = log_prob_w.mean()
        loss = - (log_prob_x + log_prob_w)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_log_prob_x', log_prob_x, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_log_prob_w', log_prob_w, on_step=False, on_epoch=True, prog_bar=False)
        return loss    

    def configure_optimizers(self):
        optimizer = torch.optim.Adamax(self.model.parameters(), **self.optimizer_kwargs)
        return optimizer
