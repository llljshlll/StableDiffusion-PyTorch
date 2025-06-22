import os
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from dataset.celeb_dataset import CelebDataset
from torch.utils.data import DataLoader
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.text_utils import *
from utils.config_utils import *
from utils.diffusion_utils import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']

    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    condition_config = diffusion_model_config['condition_config']
    assert 'text' in condition_config['condition_types'], 'Text condition required for textual inversion'

    text_tokenizer, text_model = get_tokenizer_and_model(condition_config['text_condition_config'],
                                                         ['text_embed_model'], device=device, eval_mode=False)
    token = args.token
    num_added = text_tokenizer.add_tokens([token])
    text_model.resize_token_embeddings(len(text_tokenizer))
    token_id = text_tokenizer.convert_tokens_to_ids(token)
    embedding_layer = text_model.get_input_embeddings()
    for param in text_model.parameters():
        param.requires_grad = False
    embedding_param = embedding_layer.weight[token_id]
    embedding_param.requires_grad = True

    im_dataset = CelebDataset(split='train',
                              im_path=dataset_config['im_path'],
                              im_size=dataset_config['im_size'],
                              im_channels=dataset_config['im_channels'],
                              use_latents=True,
                              latent_path=os.path.join(train_config['task_name'],
                                                       train_config['vqvae_latent_dir_name']),
                              condition_config=condition_config)

    data_loader = DataLoader(im_dataset,
                             batch_size=train_config['ldm_batch_size'],
                             shuffle=True)

    model = Unet(im_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to(device)
    if os.path.exists(args.pretrained_ckpt):
        model.load_state_dict(torch.load(args.pretrained_ckpt, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    vae = None
    if not im_dataset.use_latents:
        print('Loading vqvae model as latents not present')
        vae = VQVAE(im_channels=dataset_config['im_channels'],
                    model_config=autoencoder_model_config).to(device)
        vae.eval()
        if os.path.exists(os.path.join(train_config['task_name'],
                                       train_config['vqvae_autoencoder_ckpt_name'])):
            print('Loaded vae checkpoint')
            vae.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                        train_config['vqvae_autoencoder_ckpt_name']),
                                           map_location=device))
        else:
            raise Exception('VAE checkpoint not found and use_latents was disabled')
        for param in vae.parameters():
            param.requires_grad = False

    optimizer = Adam([embedding_param], lr=args.lr)
    criterion = torch.nn.MSELoss()
    num_epochs = args.epochs

    for epoch_idx in range(num_epochs):
        losses = []
        for im, cond_input in tqdm(data_loader):
            optimizer.zero_grad()
            im = im.float().to(device)
            if not im_dataset.use_latents:
                with torch.no_grad():
                    im, _ = vae.encode(im)

            cond_input['text'] = [t.replace(args.placeholder, token) for t in cond_input['text']]
            text_condition = get_text_representation(cond_input['text'], text_tokenizer, text_model, device)
            cond_input['text'] = text_condition

            noise = torch.randn_like(im).to(device)
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            noisy_im = scheduler.add_noise(im, noise, t)
            with torch.no_grad():
                noise_pred = model(noisy_im, t, cond_input=cond_input)
            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print('Finished epoch:{} | Loss : {:.4f}'.format(epoch_idx + 1, np.mean(losses)))
        torch.save({'token': token, 'embedding': embedding_param.detach().cpu()},
                   os.path.join(train_config['task_name'], f'textual_inversion_{token}.pth'))

    print('Done Training ...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Textual inversion fine-tuning')
    parser.add_argument('--config', dest='config_path',
                        default='config/celebhq_text_cond.yaml', type=str)
    parser.add_argument('--pretrained_ckpt', type=str, required=True,
                        help='Path to pretrained ddpm checkpoint')
    parser.add_argument('--token', type=str, required=True, help='Placeholder token to train')
    parser.add_argument('--placeholder', type=str, default='*', help='Token in captions to replace')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    train(args)
