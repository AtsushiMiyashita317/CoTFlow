import math
import torch
import pytorch_lightning as pl
import torchvision as tv
import normflows as nf
import wandb
from normflows.flows.reshape import Split as NFFSplit
from normflows.nets.cnn import ConvNet2d as NFConvNet2d

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
        num_parts=1024,
        hidden_channels=512,
        n_levels=6,
        n_blocks=32,
        eps=1e-3,
        max_log_abs_scale=0.1,
        jacobian_mode="cancel"
    ):
        super().__init__()

        self.pretrained_model = pretrained_model
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.max_log_abs_scale = max_log_abs_scale
        self.jacobian_mode = jacobian_mode

        self.num_bases = num_bases
        self.num_parts = num_parts
        self.n_levels = n_levels

        L = n_levels
        K = n_blocks
        channels = input_shape[0]

        q0 = []
        merges = []
        flows = []
        norms = []
        self.latent_shapes = []
        for i in range(L):
            flows_ = []
            for j in range(K):
                flows_ += [nf.flows.GlowBlock(
                    channels * 2 ** (L + 1 - i), 
                    hidden_channels=hidden_channels,
                    split_mode='channel', 
                    scale=True,
                    scale_map=self._scale_map
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
            norms += [nf.flows.ActNorm((latent_shape[0], 1, 1))]

        transform = nf.transforms.Loft()

        # Construct flow model with the multiscale architecture
        self.flow = nf.MultiscaleFlow(q0, flows, merges, transform)

        self.norm_cotangent = torch.nn.ModuleList(norms)

        latent_channels = [shape[0] for shape in self.latent_shapes]
        latent_channels_total = sum(latent_channels)

        A_data = torch.randn(num_parts, latent_channels_total, dtype=torch.cfloat) / latent_channels_total**0.25
        self.A = torch.nn.Parameter(A_data)
        B_data = torch.randn(num_parts, latent_channels_total, dtype=torch.cfloat) / latent_channels_total**0.25
        self.B = torch.nn.Parameter(B_data)
        C_data = torch.randn(num_bases, num_parts) / num_parts**0.5
        self.C = torch.nn.Parameter(C_data)

        lam_data = torch.randn(num_parts, input_shape[1] // 2, input_shape[2] // 4 + 1)
        self.lam = torch.nn.Parameter(lam_data)

        self.register_buffer('var_init_done', torch.tensor(0))
        self.register_buffer('var_z', torch.tensor(1.0))
        self.register_buffer('var_w', torch.tensor(1.0))
        self.register_buffer('eps', torch.tensor(eps))

    def _scale_map(self, z: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(z / self.max_log_abs_scale) * self.max_log_abs_scale
        return z.exp()

    def parameters(self, recurse = True):
        yield from self.flow.parameters(recurse)
        yield self.A
        yield self.B
        yield self.C
        yield self.lam

    def forward_model(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0)
        x = x * 2 - 1
        y = self.pretrained_model(x)
        return y.squeeze(0)

    def forward_flow(self, *z: torch.Tensor) -> torch.Tensor:
        z = [zi.unsqueeze(0) for zi in z]
        x, _ = self.flow.forward_and_log_det(z)
        x = (x + 1) / 2
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
        z_upsampled = torch.cat(z_upsampled, dim=-3)  #r (batch_size, channels_total, H, W)
        z_fft = torch.fft.rfftn(z_upsampled, dim=(-2, -1))  # (batch_size, channels_total, H, W)
        return z_fft

    def inverse_fft(self, z_fft: torch.Tensor) -> list[torch.Tensor]:
        z = torch.fft.irfftn(z_fft, dim=(-2, -1))  # (batch_size, channels_total, H, W)

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

    def forward_vector_field_half(
            self, 
            a: torch.Tensor, 
            A: torch.Tensor,
            alpha: torch.Tensor = None,
        ) -> torch.Tensor:
        """
        Args:
            a: (batch_size, channels_total, height, width)
            A: (num_parts, latent_channels_total)
            alpha: (num_parts, height, width)
        Returns:
            out: (batch_size, num_parts, height, width)
        """
        Aa = torch.einsum('mc,bchw->bmhw', A, a)  # (batch_size, num_parts, H, W)

        if alpha is not None:
            alpha = 1j * alpha.sigmoid()
            alpha[:, 0, 0] = 0.0 + 0.0j
            Aa = alpha.unsqueeze(0) * Aa  # (batch_size, num_parts, H, W)

        Aa = torch.fft.irfftn(Aa, dim=(-2, -1))  # (batch_size, num_parts, H, W)
        
        return Aa  # (batch_size, num_parts, H, W)
    
    def compute_S(
        self, 
        Az: torch.Tensor,
        Bz: torch.Tensor,
        CD: list[torch.Tensor]
    ) -> torch.Tensor:
        heights = [shape[1] for shape in self.latent_shapes]
        widths = [shape[2] for shape in self.latent_shapes]
        height_max = max(heights)
        width_max = max(widths)

        out = torch.zeros(Az.size(0), Az.size(1), Bz.size(1), device=Az.device)

        for CDi, hi, wi in zip(CD, heights, widths):
            rh = height_max // hi
            rw = width_max // wi
            Az_i = Az[..., ::rh, ::rw]  # (batch_size, num_parts, hi, wi)
            Bz_i = Bz[..., ::rh, ::rw]  # (batch_size, num_parts, hi, wi)
            out = out + torch.einsum('bmhw,bnhw->bmn', Az_i, Bz_i) * 2 * CDi.real

        return out

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
        num_parts = self.num_parts
        height_max = z[-1].size(-2)
        width_max = z[-1].size(-1)
        input_dim = self.input_shape[0] * self.input_shape[1] * self.input_shape[2]
        channels = [shape[0] for shape in self.latent_shapes]

        if self.jacobian_mode == "approx":
            w = [wi + torch.randn_like(wi) * 1e-3 ** 0.5 for wi in w]

        w_fft = self.forward_fft(*w)  # (batch_size, channels_total, H_max, W_max)
        z_fft = self.forward_fft(*z)  # (batch_size, channels_total, H_max, W_max)

        w = torch.fft.irfftn(w_fft, dim=(-2, -1))  # (batch_size, channels_total, H_max, W_max)
        S_ww = torch.einsum('bchw,bchw->b', w, w)    # (batch_size,)

        Az = self.forward_vector_field_half(z_fft, self.A, self.lam)  # (batch_size, num_parts, H_max, W_max)
        Bz = self.forward_vector_field_half(z_fft, self.B, self.lam)  # (batch_size, num_parts, H_max, W_max)
        Aw = self.forward_vector_field_half(w_fft, self.A)            # (batch_size, num_parts, H_max, W_max)
        Bw = self.forward_vector_field_half(w_fft, self.B)            # (batch_size, num_parts, H_max, W_max)

        A = torch.split(self.A, channels, dim=-1)      # list of (num_parts, channels_i)
        B = torch.split(self.B, channels, dim=-1)      # list of (num_parts, channels_i)

        AA = []
        BB = []
        AB = []

        for Ai, Bi in zip(A, B):
            AA.append(Ai @ Ai.H)    # (num_parts, num_parts)
            BB.append(Bi @ Bi.H)    # (num_parts, num_parts)
            AB.append(Ai @ Bi.H)    # (num_parts, num_parts)

        S_zz = torch.zeros(batch_size, num_parts, num_parts, device=device)
        S_zz = S_zz + self.compute_S(Az, Az, BB)
        S_zz = S_zz + self.compute_S(Bz, Bz, AA)
        tmp = self.compute_S(Bz, Az, AB)
        S_zz = S_zz + tmp + tmp.mT
        S_zz = torch.einsum('pm,bmn->bpn', self.C, S_zz)
        S_zz = torch.einsum('qn,bpn->bpq', self.C, S_zz)
        
        S_wz = torch.zeros(batch_size, num_parts, device=device)
        S_wz = S_wz + torch.einsum('bmhw,bmhw->bm', Aw, Bz)
        S_wz = S_wz + torch.einsum('bmhw,bmhw->bm', Bw, Az)
        S_wz = torch.einsum('pm,bm->bp', self.C, S_wz)

        S_wzzw = torch.einsum('bp,bp->b', S_wz, S_wz)

        if self.training:
            var_w = S_ww.mean()
            var_z = S_wzzw.mean()
            if self.var_init_done.item() == 0:
                self.var_w.fill_(var_w.detach().clone())
                self.var_z.fill_(var_z.detach().clone())
                self.var_init_done.fill_(1)
            var_w = 0.1 * var_w + 0.9 * self.var_w
            var_z = 0.1 * var_z + 0.9 * self.var_z
            self.var_w.fill_(var_w.detach().clone())
            self.var_z.fill_(var_z.detach().clone())
        else:
            var_w = self.var_w
            var_z = self.var_z

        S_ww = S_ww / var_w
        S_zz = S_zz / var_z
        S_wzzw = S_wzzw / var_z

        I = torch.eye(num_bases, device=device)
        for i in range(10):
            eps = self.eps * (2 ** i)
            M = S_zz + eps.unsqueeze(-1).unsqueeze(-1) * I               # (batch_size, num_bases, num_bases)
            try:
                L_H = torch.linalg.cholesky(M)
            except Exception:
                if i == 9:
                    raise
                continue
            break

        logdet = 2 * torch.log(torch.diagonal(L_H, dim1=-2, dim2=-1)).sum(-1)
        logdet = logdet + (input_dim - num_bases) * eps.log()   # (batch_size,)
        if self.jacobian_mode == "approx":
            logdet = logdet + S_ww.clamp_min(1e-12).log() +  (input_dim - 1) * math.log(1e-3)

        # w^t(eI + v^tv)w= e w^tw + w^tv^tvw
        trace = eps * S_ww + S_wzzw    # (batch_size,)

        logdet = logdet + math.log(input_dim) * input_dim
        trace = trace * input_dim

        log_prob = 0.5 * (logdet - trace - input_dim * math.log(2 * math.pi))

        return log_prob, trace * 0.5, logdet * 0.5

    @torch.no_grad()
    def sample(self, num_samples):
        x = self.flow.sample(num_samples)[0]
        x = (x + 1) / 2
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, logdet_x = self.flow.inverse_and_log_det(x)
        trace_x = 0.0
        for zi, q0i in zip(z, self.flow.q0):
            trace_x = trace_x + q0i.log_prob(zi)
        log_prob_x = trace_x + logdet_x  # (B,)

        if self.pretrained_model is None:
            return log_prob_x, torch.zeros_like(log_prob_x)

        w = self.sample_cotangent(x).detach()  # (B, output_dim, input_dim)

        w = self.pullback_cotangent(w, z)

        if self.jacobian_mode == "cancel":
            logdet_w = -logdet_x
        else:
            logdet_w = 0.0

        for wi, norm_i in zip(w, self.norm_cotangent):
            wi, logdet_i = norm_i.inverse(wi)
            logdet_w = logdet_w + logdet_i

        log_prob, trace, logdet = self.log_prob(w, z)  # (B,)
        log_prob_w = log_prob + logdet_w
        logdet_w = logdet
        trace_w = trace

        return log_prob_x, log_prob_w, trace_x, trace_w, logdet_x, logdet_w


class CoTGlowModule(pl.LightningModule):
    def __init__(
        self,
        pretrained_model: torch.nn.Module,
        flow_kwargs,
        sample_num=64,
        warmup_steps=1,
        beta=0.5,
        **optimizer_kwargs,
    ):
        super().__init__()

        self.model = CoTGlow(pretrained_model, **flow_kwargs)
        self.sample_num = sample_num

        self.beta = beta
        self.warmup_steps = warmup_steps
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
        log_prob_x, log_prob_w, trace_x, trace_w, logdet_x, logdet_w = self.model.forward(x)
        log_prob_x = log_prob_x.mean()
        log_prob_w = log_prob_w.mean()
        trace_x = trace_x.mean()
        trace_w = trace_w.mean()
        logdet_x = logdet_x.mean()
        logdet_w = logdet_w.mean()
        loss = - 2 * (log_prob_x * (1 - self.beta) + log_prob_w * self.beta)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_log_prob_x', log_prob_x, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_log_prob_w', log_prob_w, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_trace_x', trace_x, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_trace_w', trace_w, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_logdet_x', logdet_x, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_logdet_w', logdet_w, on_step=True, on_epoch=True, prog_bar=False)
        self.log('var_w', self.model.var_w.item(), on_step=True, on_epoch=True, prog_bar=False)
        self.log('var_z', self.model.var_z.item(), on_step=True, on_epoch=True, prog_bar=False)

        # Collect std of Split([z1, z2]) outputs in execution order and log to wandb.
        # Requirement: install/remove hooks and collect data here (line ~548).
        # split_stats: list[float] = []
        # convnet2d_stats: list[tuple[float, str]] = []
        # hook_handles: list[torch.utils.hooks.RemovableHandle] = []

        # def _split_forward_hook(_module, _inputs, output):
        #     # Split.forward returns: ([z1, z2], log_det)
        #     try:
        #         z_list = output[0]
        #         if not isinstance(z_list, (list, tuple)) or len(z_list) != 2:
        #             return
        #         z1, z2 = z_list
        #         z = torch.cat([z1, z2], dim=1)
        #         # Use population std (unbiased=False) for stability and log scalar.
        #         std = z.detach().float().std(unbiased=False).item()
        #         split_stats.append(std)
        #     except Exception:
        #         # Never break training due to debug logging.
        #         return

        # def _convnet2d_forward_hook(_module, _inputs, output):
        #     # ConvNet2d.forward returns a Tensor.
        #     try:
        #         if not torch.is_tensor(output):
        #             return
        #         y = output.detach().float()
        #         std = y.std(unbiased=False).item()
        #         convnet2d_stats.append(std)
        #     except Exception:
        #         return

        # # Attach hooks to all Split / ConvNet2d submodules under the flow.
        # for m in self.model.flow.modules():
        #     if isinstance(m, NFFSplit):
        #         hook_handles.append(m.register_forward_hook(_split_forward_hook))
        #     if isinstance(m, NFConvNet2d):
        #         hook_handles.append(m.register_forward_hook(_convnet2d_forward_hook))

        # try:
        #     _ = self.model.flow.inverse_and_log_det(x)
        # finally:
        #     for h in hook_handles:
        #         try:
        #             h.remove()
        #         except Exception:
        #             pass

        # if split_stats:
        #     wandb_logger = self.logger
        #     if hasattr(wandb_logger, "experiment"):
        #         table = wandb.Table(columns=["step", "call_index", "std"])
        #         for i, std in enumerate(split_stats):
        #             table.add_data(self.global_step, i, std)
        #         wandb_logger.experiment.log({"debug/split_output_std_table": table,})

        # if convnet2d_stats:
        #     wandb_logger = self.logger
        #     if hasattr(wandb_logger, "experiment"):
        #         table = wandb.Table(
        #             columns=["step", "call_index", "std"]
        #         )
        #         for i, std in enumerate(convnet2d_stats):
        #             table.add_data(self.global_step, i, std)
        #         wandb_logger.experiment.log({"debug/convnet2d_output_std_table": table,})

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        log_prob_x, log_prob_w, trace_x, trace_w, logdet_x, logdet_w = self.model.forward(x)
        log_prob_x = log_prob_x.mean()
        log_prob_w = log_prob_w.mean()
        trace_x = trace_x.mean()
        trace_w = trace_w.mean()
        logdet_x = logdet_x.mean()
        logdet_w = logdet_w.mean()
        loss = - 2 * (log_prob_x * (1 - self.beta) + log_prob_w * self.beta)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_log_prob_x', log_prob_x, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_log_prob_w', log_prob_w, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_trace_x', trace_x, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_trace_w', trace_w, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_logdet_x', logdet_x, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_logdet_w', logdet_w, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adamax(self.model.parameters(), **self.optimizer_kwargs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min((step + 1) / self.warmup_steps, 1.0)
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }
