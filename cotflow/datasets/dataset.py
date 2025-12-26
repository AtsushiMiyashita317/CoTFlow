import torch
import torchvision as tv
import h5py


class DatasetMNIST(torch.utils.data.Dataset):
    def __init__(self, path, train=True, enable_cache=False):
        # load dataset
        self.dataset = tv.datasets.MNIST(
            root=path, 
            train=train, 
            download=True, 
            transform=tv.transforms.ToTensor()
        )
        self.enable_cache = enable_cache
        if self.enable_cache:
            self.cached_data = [None] * len(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.enable_cache and self.cached_data[idx] is not None:
            return self.cached_data[idx]

        img, label = self.dataset[idx]
        data = dict(
            image=img,
            label=torch.tensor(label)
        )
        if self.enable_cache:
            self.cached_data[idx] = data
        return data

class Dataset3DShapes(torch.utils.data.Dataset):
    def __init__(self, path):
        # load dataset
        dataset = h5py.File(path, 'r')
        self.images = dataset['images'][:]
        self.labels = dataset['labels'][:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img = img / 255.0

        label = self.labels[idx]
        label = torch.from_numpy(label).float()
        color = label[0:3].mul(2 * torch.pi)
        color = torch.stack([color.sin(), color.cos()], dim=-1)  # (3, 2)
        scale = label[3:4]
        shape = label[4].long()
        orientation = label[5].div(30)
        return dict(
            image=img,
            color=color,
            scale=scale,
            shape=shape,
            orientation=orientation
        )


class DatasetCelebA(torch.utils.data.Dataset):
    def __init__(
        self, 
        path, 
        split='all', 
        image_size=(218, 178), 
        enable_cache=False,
        bits=5,
    ):
        self.data = tv.datasets.CelebA(
            root=path, 
            split=split,
            target_type=['attr', 'identity', 'bbox', 'landmarks'], 
            download=False, 
            transform=tv.transforms.Compose([
                tv.transforms.Resize(image_size),
                tv.transforms.ToTensor()
            ])
        )
        self.bits = bits
        self.enable_cache = enable_cache
        if self.enable_cache:
            self.cached_data = [None] * len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.enable_cache and self.cached_data[idx] is not None:
            return self.cached_data[idx]

        img, (attr, identity, bbox, landmarks) = self.data[idx]
        img = img * 255.0
        img = torch.floor(img / 2 ** (8 - self.bits))
        img = img + torch.rand_like(img)  # dequantization
        img = img / 2 ** self.bits  # rescale to [0,1]
        bbox = torch.stack(
            (bbox[0] / 178, bbox[1] / 218,
            (bbox[0] + bbox[2]) / 178, (bbox[1] + bbox[3]) / 218)
        )
        landmarks = landmarks.view(-1, 2) / torch.tensor([178.0, 218.0])
        landmarks = landmarks.flatten()
        data = dict(
            image=img,
            attr=(attr + 1) // 2,  # -1,1 -> 0,1
            identity=identity - 1,
            bbox=bbox,
            landmark=landmarks
        )
        if self.enable_cache:
            self.cached_data[idx] = data
        return data
    

class DatasetDumpedFeatures(torch.utils.data.Dataset):
    def __init__(
        self, 
        h5_path, 
        split='train', 
        enable_cache=False,
        target_type=['attr']
    ):
        self.h5 = h5py.File(h5_path, 'r')
        self.z = self.h5[split]['z']
        self.jac_predictors = {}
        for name in target_type:
            dataset_name = f'jac_predictor_{name}'
            self.jac_predictors[name] = self.h5[split][dataset_name]
        self.enable_cache = enable_cache
        if self.enable_cache:
            self.cached_data = [None] * len(self.z)

    def __len__(self):
        return self.z.shape[0]

    def __getitem__(self, idx):
        if self.enable_cache and self.cached_data[idx] is not None:
            return self.cached_data[idx]
        z = torch.from_numpy(self.z[idx])
        jacs = [torch.from_numpy(dataset[idx]) for dataset in self.jac_predictors.values()]
        jacs = torch.cat(jacs, dim=0)
        if self.enable_cache:
            self.cached_data[idx] = (z, jacs)
        return z, jacs
