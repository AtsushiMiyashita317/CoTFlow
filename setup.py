from setuptools import setup, find_packages

setup(
    name="cotflow",
    version="0.1.0",
    description="Conditional Flow-based generative models with autoencoder for CelebA and 3D Shapes datasets.",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "pytorch-lightning",
        "tqdm",
        "gdown",
        "wandb",
        "hydra-core",
        "omegaconf",
        "h5py",
    ],
    entry_points={
        "console_scripts": [
            "train_autoencoder=cotflow.bin.train_autoencoder:main",
            "train_predictor=cotflow.bin.train_predictor:main",
        ],
    },
    include_package_data=True,
    python_requires=">=3.8",
)
