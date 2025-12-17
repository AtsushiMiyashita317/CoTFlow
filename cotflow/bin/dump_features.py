import os
import argparse
import torch
import h5py
from tqdm import tqdm
import yaml

import cotflow.nets.autoencoder as autoencoder
from cotflow.nets.autoencoder import AutoencoderModule
import cotflow.nets.predictor as predictor
from cotflow.nets.predictor import PredictorLightningModule
import cotflow.datasets as datasets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--autoencoder', type=str, required=True)
    parser.add_argument('--predictors', type=str, nargs='+', required=True)
    parser.add_argument('--dump_h5', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    ckpt_path = os.path.join(args.autoencoder, "model.ckpt")
    config_path = os.path.join(args.autoencoder, "config.yaml")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    autoencoder_module = AutoencoderModule.load_from_checkpoint(
        ckpt_path, model_name=config['model']['name'], model_args=config['model']['args'])
    autoencoder_model = autoencoder_module.model
    autoencoder_model.eval().to(args.device)

    train_dataset = getattr(datasets, config['train']['dataset']['name'])(**config['train']['dataset']['args'])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    val_dataset = getattr(datasets, config['val']['dataset']['name'])(**config['val']['dataset']['args'])
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    predictors = {}
    for p in args.predictors:
        name, path = p.split('=')
        ckpt_path = os.path.join(path, "model.ckpt")
        config_path = os.path.join(path, "config.yaml")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        predictor_model = getattr(predictor, config['model']['name'])(**config['model']['args'])
        predictor_module = PredictorLightningModule.load_from_checkpoint(
            ckpt_path, autoencoder=autoencoder_model,
            model_name=config['model']['name'], model_args=config['model']['args'])
        predictor_model = predictor_module.model
        predictor_model.eval().to(args.device)
        predictors[name] = predictor_model

    # h5ファイル作成
    N_train = len(train_dataset)
    N_val = len(val_dataset)
    # 1サンプルでshape推定
    with torch.no_grad():
        x0 = next(iter(train_dataloader))['image'].to(args.device)
        z0 = autoencoder_model.encode(x0)
        x_recon0 = autoencoder_model.decode(z0)
        pred0 = {k: p(x_recon0) for k, p in predictors.items()}

    latent_dim = z0.shape[1]
    h5 = h5py.File(args.dump_h5, 'w')
    g_train = h5.create_group('train')
    g_val = h5.create_group('val')
    g_train.create_dataset('z', shape=(N_train, latent_dim), dtype='f4')
    for k, v in pred0.items():
        g_train.create_dataset(f'jac_predictor_{k}', shape=(N_train, v.shape[1], latent_dim), dtype='f4')
    g_val.create_dataset('z', shape=(N_val, latent_dim), dtype='f4')
    for k, v in pred0.items():
        g_val.create_dataset(f'jac_predictor_{k}', shape=(N_val, v.shape[1], latent_dim), dtype='f4')

    for dataloader, g in [(train_dataloader, g_train), (val_dataloader, g_val)]:
        idx = 0
        for batch in tqdm(dataloader):
            x = batch['image'].to(args.device)
            with torch.no_grad():
                z = autoencoder_model.encode(x)
            z = z.detach()
            jac_pred = {}
            for k, p in predictors.items():
                jac_pred[k] = torch.func.vmap(torch.func.jacrev(
                    lambda z: p(autoencoder_model.decode(z)))
                )(z.unsqueeze(1)).reshape(x.shape[0], -1, latent_dim)

            bsz = x.shape[0]
            g['z'][idx:idx+bsz] = z.detach().cpu().numpy()
            for k, v in jac_pred.items():
                g[f'jac_predictor_{k}'][idx:idx+bsz] = v.detach().cpu().numpy()
            idx += bsz

    h5.close()

if __name__ == '__main__':
	main()
