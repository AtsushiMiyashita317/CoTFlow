import torch
import torchvision as tv
import h5py


class DatasetMNIST(torch.utils.data.Dataset):
    def __init__(self, path, train=True):
        # load dataset
        self.dataset = tv.datasets.MNIST(
            root=path, 
            train=train, 
            download=True, 
            transform=tv.transforms.ToTensor()
        )   

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return dict(
            image=img,
            label=torch.tensor(label)
        )

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
    def __init__(self, path, split='all', image_size=(218, 178)):
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, (attr, identity, bbox, landmarks) = self.data[idx]
        bbox = torch.stack(
            (bbox[0] / 178, bbox[1] / 218,
            (bbox[0] + bbox[2]) / 178, (bbox[1] + bbox[3]) / 218)
        )
        landmarks = landmarks.view(-1, 2) / torch.tensor([178.0, 218.0])
        landmarks = landmarks.flatten()
        return dict(
            image=img,
            attr=(attr + 1) // 2,  # -1,1 -> 0,1
            identity=identity - 1,
            bbox=bbox,
            landmark=landmarks
        )
