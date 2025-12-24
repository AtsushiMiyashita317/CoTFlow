import sys
import pytorch_lightning as pl
import torch
import torch.nn as nn

# 共通LightningModule
class PredictorLightningModule(pl.LightningModule):
    def __init__(
        self, 
        model_name, 
        model_args,
        autoencoder=None, 
        task='attr', 
        lr=1e-3
    ):
        """
        model: nn.Module (出力次元で自動判別)
        task: 'attr', 'identity', 'bbox', 'landmarks'
        """
        super().__init__()
        self.model = getattr(sys.modules[__name__], model_name)(**model_args)
        self.autoencoder = autoencoder
        self.task = task
        self.lr = lr
        if task == 'attr':
            self.criterion = nn.BCEWithLogitsLoss()
        elif task == 'identity':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

    def forward(self, x):
        if self.autoencoder is not None:
            with torch.no_grad():
                _, x = self.autoencoder(x)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch['image']
        y = batch[self.task]
        logits = self(x)
        if self.task == 'identity':
            loss = self.criterion(logits, y.long())
            acc = (logits.argmax(dim=1) == y).float().mean()
            self.log('train_acc', acc, prog_bar=True)
        elif self.task == 'attr':
            loss = self.criterion(logits, y.float())
            pred = (logits.sigmoid() > 0.5).float()
            acc = (pred == y).float().mean()
            self.log('train_acc', acc, prog_bar=True)
        else:
            loss = self.criterion(logits, y.float())
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y = batch[self.task]
        logits = self(x)
        if self.task == 'identity':
            loss = self.criterion(logits, y.long())
            acc = (logits.argmax(dim=1) == y).float().mean()
            self.log('val_acc', acc, prog_bar=True)
        elif self.task == 'attr':
            loss = self.criterion(logits, y.float())
            pred = (logits.sigmoid() > 0.5).float()
            acc = (pred == y).float().mean()
            self.log('val_acc', acc, prog_bar=True)
        else:
            loss = self.criterion(logits, y.float())
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# 属性予測器（40次元属性）
class CelebAAttrPredictor(nn.Module):
	"""
	入力: 3x218x178画像 (CelebAフルサイズ)
	出力: 40次元属性ベクトル
	"""
	def __init__(self, in_channels=3, num_attrs=40, input_shape=(3, 218, 178)):
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(32),
			nn.Softplus(),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.Softplus(),
			nn.MaxPool2d(2),
			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128),
			nn.Softplus(),
			nn.MaxPool2d(2),
			nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(256),
			nn.Softplus(),
			nn.MaxPool2d(2),
		)
		# 動的に全結合層の入力次元を計算
		with torch.no_grad():
			dummy = torch.zeros(1, *input_shape)
			feat = self.features(dummy)
			flatten_dim = feat.view(1, -1).shape[1]
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(flatten_dim, 512),
			nn.Softplus(),
			nn.Dropout(0.3),
			nn.Linear(512, num_attrs),
		)

	def forward(self, x):
		feat = self.features(x)
		out = self.classifier(feat)
		return out


class CelebAIdentityPredictor(nn.Module):
	"""
	入力: 3x218x178画像 (CelebAフルサイズ)
	出力: 10177クラス分類
	"""
	def __init__(self, in_channels=3, num_classes=10177, input_shape=(3, 218, 178)):
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(32),
			nn.Softplus(),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.Softplus(),
			nn.MaxPool2d(2),
			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128),
			nn.Softplus(),
			nn.MaxPool2d(2),
			nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(256),
			nn.Softplus(),
			nn.MaxPool2d(2),
		)
		# 動的に全結合層の入力次元を計算
		with torch.no_grad():
			dummy = torch.zeros(1, *input_shape)
			feat = self.features(dummy)
			flatten_dim = feat.view(1, -1).shape[1]
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(flatten_dim, 512),
			nn.Softplus(),
			nn.Dropout(0.3),
			nn.Linear(512, num_classes),
		)

	def forward(self, x):
		feat = self.features(x)
		out = self.classifier(feat)
		return out


# bbox予測器（4次元回帰）
class CelebABBoxPredictor(nn.Module):
	"""
	入力: 3x218x178画像 (CelebAフルサイズ)
	出力: 4次元バウンディングボックス (x, y, w, h)
	"""
	def __init__(self, in_channels=3, out_dim=4, input_shape=(3, 218, 178)):
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(32),
			nn.Softplus(),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.Softplus(),
			nn.MaxPool2d(2),
			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128),
			nn.Softplus(),
			nn.MaxPool2d(2),
			nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(256),
			nn.Softplus(),
			nn.MaxPool2d(2),
		)
		with torch.no_grad():
			dummy = torch.zeros(1, *input_shape)
			feat = self.features(dummy)
			flatten_dim = feat.view(1, -1).shape[1]
		self.regressor = nn.Sequential(
			nn.Flatten(),
			nn.Linear(flatten_dim, 256),
			nn.Softplus(),
			nn.Linear(256, out_dim),
		)

	def forward(self, x):
		feat = self.features(x)
		out = self.regressor(feat)
		return out


# landmarks予測器（5点×2=10次元回帰）
class CelebALandmarkPredictor(nn.Module):
	"""
	入力: 3x218x178画像 (CelebAフルサイズ)
	出力: 10次元ランドマーク座標
	"""
	def __init__(self, in_channels=3, out_dim=10, input_shape=(3, 218, 178)):
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(32),
			nn.Softplus(),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.Softplus(),
			nn.MaxPool2d(2),
			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128),
			nn.Softplus(),
			nn.MaxPool2d(2),
			nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(256),
			nn.Softplus(),
			nn.MaxPool2d(2),
		)
		with torch.no_grad():
			dummy = torch.zeros(1, *input_shape)
			feat = self.features(dummy)
			flatten_dim = feat.view(1, -1).shape[1]
		self.regressor = nn.Sequential(
			nn.Flatten(),
			nn.Linear(flatten_dim, 256),
			nn.Softplus(),
			nn.Linear(256, out_dim),
		)

	def forward(self, x):
		feat = self.features(x)
		out = self.regressor(feat)
		return out
