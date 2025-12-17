import torch
import torchvision as tv
import pytorch_lightning as pl
import wandb

class AbsAutoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_dim = None
        self.encoder: torch.nn.Module = None
        self.decoder: torch.nn.Module = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon


class Autoencoder(AbsAutoencoder):
    def __init__(
        self,
        input_shape,
        channel_multiplier=0.5,
        num_layers=4,
    ):
        super().__init__()
        layers = []
        layers.append(torch.nn.Flatten())

        input_channels, input_height, input_width = input_shape
        current_channels = input_channels * input_height * input_width

        channel_list = self._calculate_parameters(current_channels, channel_multiplier)
        channel_list = [current_channels] + channel_list[:num_layers]

        for channel in channel_list[1:]:
            layers.extend([
                torch.nn.Linear(current_channels, channel),
                torch.nn.Softplus(),
                torch.nn.LayerNorm([channel]),
            ])
            current_channels = channel

        self.encoder = torch.nn.Sequential(*layers)
        self.latent_dim = current_channels

        layers = []
        for channel in reversed(channel_list[:-1]):
            layers.extend([
                torch.nn.Softplus(),
                torch.nn.LayerNorm([current_channels]),
                torch.nn.Linear(current_channels, channel),
            ])
            current_channels = channel
        layers.append(torch.nn.Sigmoid())
        layers.append(torch.nn.Unflatten(1, tuple(input_shape)))
        self.decoder = torch.nn.Sequential(*layers)

    @staticmethod
    def _calculate_parameters(input_dim, channel_multiplier):
        channel_list = []
        current_dim = input_dim
        while current_dim * channel_multiplier >= 1:
            current_dim = int(current_dim * channel_multiplier)
            channel_list.append(current_dim)
        return channel_list


class Autoencoder2D(AbsAutoencoder):
    def __init__(
        self, 
        input_shape,
        base_kernel_size=2,
        stride=2,
        channel_multiplier=2,
        num_layers=10,
    ):
        super().__init__()
        input_channels, input_height, input_width = input_shape

        kernel_size_list_h, output_size_list_h = self._calculate_conv_parameters(input_height, base_kernel_size, stride)
        kernel_size_list_w, output_size_list_w = self._calculate_conv_parameters(input_width, base_kernel_size, stride)
        stride_list_h = [stride] * len(kernel_size_list_h)
        stride_list_w = [stride] * len(kernel_size_list_w)

        if len(kernel_size_list_h) > len(kernel_size_list_w):
            kernel_size_list_w.extend([1] * (len(kernel_size_list_h) - len(kernel_size_list_w)))
            output_size_list_w.extend([output_size_list_w[-1]] * (len(output_size_list_h) - len(output_size_list_w)))
            stride_list_w.extend([1] * (len(output_size_list_h) - len(stride_list_w)))
        elif len(kernel_size_list_w) > len(kernel_size_list_h):
            kernel_size_list_h.extend([1] * (len(kernel_size_list_w) - len(kernel_size_list_h)))
            output_size_list_h.extend([output_size_list_h[-1]] * (len(output_size_list_w) - len(output_size_list_h)))
            stride_list_h.extend([1] * (len(output_size_list_w) - len(stride_list_h)))

        if num_layers <= len(kernel_size_list_h):
            kernel_size_list_h = kernel_size_list_h[:num_layers]
            kernel_size_list_w = kernel_size_list_w[:num_layers]
            output_size_list_h = output_size_list_h[:num_layers]
            output_size_list_w = output_size_list_w[:num_layers]
            stride_list_h = stride_list_h[:num_layers]
            stride_list_w = stride_list_w[:num_layers]

        output_height = output_size_list_h[-1] if output_size_list_h else input_height
        output_width = output_size_list_w[-1] if output_size_list_w else input_width

        conv_channel_list = [input_channels]
        current_channels = input_channels
        for _ in kernel_size_list_h:
            current_channels = int(current_channels * channel_multiplier)
            conv_channel_list.append(current_channels)

        current_channels = current_channels * output_height * output_width

        if num_layers > len(kernel_size_list_h):
            linear_channel_list = self._calculate_linear_parameters(current_channels, stride, channel_multiplier)
            linear_channel_list = [current_channels] + linear_channel_list[:num_layers - len(kernel_size_list_h)]
        else:
            linear_channel_list = [current_channels]

        self.latent_dim = current_channels

        layers = []
        current_channels = input_channels

        for k_h, k_w, o_h, o_w, s_h, s_w, c in zip(
            kernel_size_list_h, kernel_size_list_w, output_size_list_h, output_size_list_w, stride_list_h, stride_list_w, conv_channel_list[1:]
        ):
            layers.extend([
                torch.nn.Conv2d(current_channels, c, kernel_size=(k_h, k_w), stride=(s_h, s_w)),
                torch.nn.Softplus(),
                torch.nn.LayerNorm([c, o_h, o_w]),
            ])
            current_channels = c

        layers.append(torch.nn.Flatten())
        current_channels = current_channels * output_height * output_width

        for channel in linear_channel_list[1:]:
            layers.extend([
                torch.nn.Linear(current_channels, channel),
                torch.nn.Softplus(),
                torch.nn.LayerNorm([channel]),
            ])
            current_channels = channel

        self.encoder = torch.nn.Sequential(*layers)
        self.latent_dim = current_channels

        layers = []

        for channel in reversed(linear_channel_list[:-1]):
            layers.extend([
                torch.nn.Softplus(),
                torch.nn.LayerNorm([current_channels]),
                torch.nn.Linear(current_channels, channel),
            ])
            current_channels = channel

        layers.append(torch.nn.Unflatten(1, (current_channels // (output_height * output_width), output_height, output_width)))
        current_channels = current_channels // (output_height * output_width)

        for k_h, k_w, o_h, o_w, s_h, s_w, c in reversed(list(zip(
            kernel_size_list_h, kernel_size_list_w, output_size_list_h, output_size_list_w, stride_list_h, stride_list_w, conv_channel_list[:-1]
        ))):
            layers.extend([
                torch.nn.Softplus(),
                torch.nn.LayerNorm([current_channels, o_h, o_w]),
                torch.nn.ConvTranspose2d(current_channels, c, kernel_size=(k_h, k_w), stride=(s_h, s_w)),
            ])
            current_channels = c

        layers.append(torch.nn.Sigmoid())

        self.decoder = torch.nn.Sequential(*layers)

    @staticmethod
    def _calculate_conv_parameters(input_dim, base_kernel_size, stride):
        kernel_size_list = []
        output_size_list = []
        current_dim = input_dim
        while current_dim >= base_kernel_size:
            res = (current_dim - base_kernel_size) % stride
            kernel_size = base_kernel_size + res
            kernel_size_list.append(kernel_size)
            current_dim = (current_dim - kernel_size) // stride + 1
            output_size_list.append(current_dim)
        if current_dim > 1:
            kernel_size_list.append(current_dim)
            output_size_list.append(1)
        return kernel_size_list, output_size_list
    
    @staticmethod
    def _calculate_linear_parameters(input_dim, stride, channel_multiplier):
        channel_list = []
        current_dim = input_dim
        while current_dim * channel_multiplier >= (stride ** 2):
            current_dim = int(current_dim * channel_multiplier / (stride ** 2))
            channel_list.append(current_dim)
        return channel_list


class AutoencoderModule(pl.LightningModule):
    def __init__(
        self, 
        model: AbsAutoencoder, 
        sample_num=64,
        loss_fn="mse",
        **optimizer_config
    ):
        super().__init__()
        self.model = model
        self.recon_imgs = None
        self.original_imgs = None
        self.sample_num = sample_num
        self.loss_fn = loss_fn
        self.optimizer_config = optimizer_config
            
    def on_validation_epoch_end(self):
        # 検証終了時に画像生成しwandbに記録
        if self.recon_imgs is None:
            return
        with torch.no_grad():
            img = torch.cat([self.original_imgs[:self.sample_num], self.recon_imgs[:self.sample_num]], dim=-1)
            grid = tv.utils.make_grid(img.cpu(), nrow=8)
            wandb_logger = self.logger
            if hasattr(wandb_logger, "experiment"):
                wandb_logger.experiment.log(
                    {f"image/reconstruction": wandb.Image(grid, caption=f"epoch {self.current_epoch}")})
            self.recon_imgs = None

    def encode(self, x):
        return self.model.encode(x)

    def decode(self, z):
        return self.model.decode(z)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        recon = self.model(batch['image'])[1]
        if self.loss_fn == "mse":
            loss = torch.nn.functional.mse_loss(recon, batch['image'])
        elif self.loss_fn == "bce":
            loss = torch.nn.functional.binary_cross_entropy(recon, batch['image'])
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_fn}")
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        recon = self.model(batch['image'])[1]
        if self.recon_imgs is None:
            self.original_imgs = batch['image'].clamp(0, 1).cpu()
            self.recon_imgs = recon.clamp(0, 1).cpu()
        if self.loss_fn == "mse":
            loss = torch.nn.functional.mse_loss(recon, batch['image'])
        elif self.loss_fn == "bce":
            loss = torch.nn.functional.binary_cross_entropy(recon, batch['image'])
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_fn}")
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adamax(self.model.parameters(), **self.optimizer_config)
        return optimizer
