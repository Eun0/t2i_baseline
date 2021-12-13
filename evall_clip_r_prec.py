from __future__ import print_function

import os
import sys
dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

from miscc.utils import mkdir_p,truncated_z_sample
from miscc.config import cfg, cfg_from_file
from model import NetG
from datasets import TextDataset, prepare_data

from DAMSM import RNN_ENCODER,CNN_ENCODER

import random
import argparse
import numpy as np
from PIL import Image
import glob

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from generator_backbones import DF_GEN
import torchvision.utils as vutils
from tqdm import tqdm
from collections import OrderedDict
from eval.fid_score import calculate_fid_given_paths
import clip


import multiprocessing
multiprocessing.set_start_method('spawn', True)

UPDATE_INTERVAL = 200
def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument("--cfg", dest="cfg_file", type=str, default="cfg/coco_df.yml")
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--manualSeed', type=int, help='manual seed',default=100)
    parser.add_argument('--num_samples',type=int,default=30000)
    parser.add_argument('--metric',type=str,default='both')
    parser.add_argument('--model_dir',type=str,default='/work/capD/exps/df256_capGD')
    parser.add_argument('--img_size',type=int,default=256)
    parser.add_argument('--batch_size', type=int, default=24)
    args = parser.parse_args()
    return args

def seed_worker(worker_id):
    worker_seed = torch.initial_seed()%2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def sampling(netG, batch_size, dataset, num_samples, model_dir):

    #print(model_dir)
    #model_list = sorted(glob.glob(f'{model_dir}/*.pth'), key=lambda x: int(''.join(filter(str.isdigit, x))))[-1:]
    #model_list = f"{model_dir}/checkpoint_"
    model_list = [f"/home/eun0/Baselines/DF-GAN/models/coco/netG_120.pth"] 
    print(f'model_list : {model_list}')
    save_dir = model_dir.split("exps")[-1]
    
    log_dir = f'../logs/{save_dir}'

    results = {'r_epoch':0,'R_mean':0,'f_epoch':0,'fid':1000}
    ixtoword = dataset.ixtoword

    os.makedirs(log_dir,exist_ok=True)

    encoder, preprocess = clip.load("ViT-B/32", device="cuda")
    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=256)
    state_dict = torch.load(cfg.TEXT.ENCODER_NAME, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    text_encoder.eval()
    text_encoder.cuda()

    for model in model_list:
        checkpoint = torch.load(model, map_location="cpu")
        netG.load_state_dict(checkpoint)
    
        netG.eval()
        netG.cuda()
        result_dir = f'../test/{save_dir}'
        mkdir_p(result_dir)

        folder = f'{result_dir}'
        mkdir_p(folder)

        cnt = 0
        R_count = 0 
        R = np.zeros(num_samples)
        cont = True
        
        g = torch.Generator()
        g.manual_seed(100)
        dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True,
        #shuffle=False, num_workers=1)
        shuffle=False, generator=g, worker_init_fn=seed_worker, num_workers=1)


        for data in dataloader:
            if cont == False:
                break

            imgs,captions,cap_lens,class_ids,keys = prepare_data(data)
            hidden = text_encoder.init_hidden(captions.size(0))
            words_embs, sent_embs = text_encoder(captions, cap_lens, hidden)
            words_embs, sent_embs = words_embs.detach(), sent_embs.detach()

            noise = truncated_z_sample(sent_embs.size(0), 100,seed=100)
            noise = torch.from_numpy(noise).float().cuda()
            if "coco_df" in args.cfg_file:
                fake_imgs = netG(noise, sent_embs)
            elif "coco_cap" in args.cfg_file:
                fake_imgs = netG(noise, sent_embs)
            elif "coco_attn" in args.cfg_file:
                raise NotImplementedError
            elif "coco_dm" in args.cfg_file:
                raise NotImplementedError
        
            clip_images = []
            clip_texts = []

            for j in range(sent_embs.size(0)):
                im = fake_imgs[j].data.cpu().numpy()
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)
                sent = ' '.join([ixtoword[ix.item()] for ix in captions[j] if ix.item()!=0])
                #fullpath = f'{folder}/{sent}_{keys[j]}.png'
                #im.save(fullpath)
                #image = preprocess(im).unsqueeze(0).cuda()
                #real_im = imgs[0][j].data.cpu().numpy()
                #real_im = (real_im + 1.0) * 127.5
                #real_im = real_im.astype(np.uint8)
                #real_im = np.transpose(real_im, (1,2,0))
                #real_im = Image.fromarray(real_im)
                #image = preprocess(real_im).unsqueeze(0).cuda()
                image = preprocess(im).unsqueeze(0).cuda()
                text = clip.tokenize(sent).cuda()
                clip_images.append(image)
                clip_texts.append(text)
                cnt += 1

            if cnt >= num_samples:
                cont = False

            s_r = ''
            s_fid = ''

            #img_features = torch.tensor(img_features)
            #text_features = torch.tensor(text_features)

            for i in range(batch_size):
                mis_captions, mis_captions_len = dataset.get_mis_caption(class_ids[i])
                mis_captions = mis_captions.tolist()
                mis_captions = [' '.join([ixtoword[ix] for ix in mis_cap if ix != 0]) for mis_cap in mis_captions]
                mis_captions = clip.tokenize(mis_captions).cuda()
                text = torch.cat((clip_texts[i], mis_captions))
                logits_per_image, _ = encoder(clip_images[i], text)
                scores0 = logits_per_image.softmax(dim=-1)
                if torch.argmax(scores0) == 0:
                    R[R_count] = 1
                R_count += 1
            

            if R_count >= num_samples:
                sum = np.zeros(10)
                np.random.shuffle(R)
                assert num_samples%10 == 0

                for i in range(10):
                    sum[i] = np.average(R[i * int(num_samples//10):(i + 1) * int(num_samples//10) - 1])
                R_mean = np.average(sum)
                R_std = np.std(sum)

                s_r = f' R mean:{R_mean:.4f} std:{R_std:.4f} '
                
                if results['R_mean'] < R_mean:
                    results['r_epoch'] = 120 
                    results['R_mean'] = R_mean

                    
            # if cnt>=num_samples:
            #     paths=["",""]
            #     paths[0] = f'eval/coco_val.npz'
            #     paths[1] = f'{folder}/'
            #     fid_value = calculate_fid_given_paths(paths, 50, True, 2048)
            #     s_fid = f'FID: {fid_value}'
            #     if fid_value < results['fid']:
            #         results['f_epoch'] = iteration 
            #         results['fid'] = fid_value

            if cnt >= num_samples:
                #s = f'epoch : {iteration} {s_r} {s_fid}'
                s = f"epoch : 120 {s_r}"
                print(s)
                with open(f'{log_dir}/log.txt','a+') as f:
                    f.write(s+'\n')

    s_res = f"Best models is {results['r_epoch']} with R mean : {results['R_mean']} {results['f_epoch']} with fid : {results['fid']}"
    print(s_res)
    with open(f'{log_dir}/log.txt','a+') as f:
        f.write(s_res+'\n')
        
if __name__ == "__main__":
    args = parse_args()
    assert args.num_samples % args.batch_size == 0 

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id


    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.deterministic = True
    cudnn.benchmark = False

    print("seed now is : ",args.manualSeed)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    torch.cuda.manual_seed_all(args.manualSeed)

    # Get data loader ##################################################
    
    imsize = args.img_size 
    batch_size = args.batch_size 
    image_transform = transforms.Compose([
            transforms.Resize((imsize,imsize)),
        ])
    print(f'Generate {imsize}x{imsize} images')
    dataset = TextDataset(cfg.DATA_DIR, 'test',
                            base_size=imsize,
                            transform=image_transform)
    assert dataset
    print(f'dataset size : {len(dataset)}')
    

    #netG = DF_GEN(visual_feature_size=32, noise_size=100, img_size=args.img_size, cond_size=256)
    if "coco_df" in args.cfg_file:
        netG = NetG(32, 100, cfg.TEXT.EMBEDDING_DIM).cuda()
    elif "coco_cap" in args.cfg_file:
        netG = DF_GEN(visual_feature_size=32, noise_size=100, img_size=args.img_size, cond_size=256)
    elif "coco_attn" in args.cfg_file:
        raise NotImplementedError
    elif "coco_dm" in args.cfg_file:
        raise NotImplementedError
    
    #text_encoder = RNN_ENCODER(dataset.n_words, nhidden=256)
    #state_dict = torch.load(cfg.TEXT.ENCODER_NAME, map_location=lambda storage, loc: storage)
    #text_encoder.load_state_dict(state_dict)
    #text_encoder.eval()
    #text_encoder.cuda()
    #text_encoder = 
    #image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    #image_encoder.load_state_dict(torch.load(cfg.TEXT.ENCODER_NAME.replace('text','image'),map_location='cpu'))
    #image_encoder.eval()
    #image_encoder.cuda()

    with torch.no_grad(): 
        sampling(netG, batch_size, dataset, args.num_samples, cfg.TRAIN.NET_G)  # generate images for the whole valid dataset
        


        
