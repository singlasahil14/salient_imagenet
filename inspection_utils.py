import pandas as pd
import numpy as np
import os
from collections import defaultdict

from PIL import Image

import torch as ch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

import clip
import timm

class NoiseSensitivityAnalysis:
    def __init__(self, images_path, masks_path, batch_size=32):
        self.images_path = images_path
        self.masks_path = masks_path
        self.batch_size = batch_size

    def add_gaussian_noise(self, images, masks, noise_mean, noise_std):
        gaussian_noise = noise_mean + noise_std * ch.randn(*images.shape).to(images.device)
        
        images_n = images + (gaussian_noise * masks)
        images_n = ch.clamp(images_n, 0., 1.)
        return images_n
        
    def compute_noisy_acc(self, inspection_model, class_index, feature_indices, 
                          noise_mean=0., noise_std=0.25):
        preds_all = []
        preds_noisy_all = []
        
        salient_imagenet = SalientImageNet(self.images_path, self.masks_path, class_index, feature_indices)        
        data_loader = DataLoader(salient_imagenet, batch_size=self.batch_size, shuffle=False)
        
        for images_batch, masks_batch in data_loader:
            
            images_batch, masks_batch = images_batch.cuda(), masks_batch.cuda()
            with ch.no_grad():
                logits_batch = inspection_model(images_batch)
            preds_batch = ch.argmax(logits_batch, dim=1).cpu().numpy()            
            preds_all.append(preds_batch)

            images_noisy_batch = self.add_gaussian_noise(images_batch, masks_batch, 
                                                         noise_mean=0., noise_std=noise_std)
            images_noisy_batch = images_noisy_batch.float().cuda()
            with ch.no_grad():
                logits_noisy_batch = inspection_model(images_noisy_batch)
            preds_noisy_batch = ch.argmax(logits_noisy_batch, dim=1).cpu().numpy()
            preds_noisy_all.append(preds_noisy_batch)

        preds_all = np.concatenate(preds_all, axis=0)
        preds_noisy_all = np.concatenate(preds_noisy_all, axis=0)
        
        clean_acc = np.sum(preds_all == class_index)/len(preds_all)
        noisy_acc = np.sum(preds_noisy_all == class_index)/len(preds_all)
        return clean_acc, noisy_acc
    
    
class SalientImageNet(Dataset):
    def __init__(self, images_path, masks_path, class_index, feature_indices, 
                 resize_size=256, crop_size=224):
        self.transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor()
        ])

        wordnet_dict = eval(open(os.path.join(masks_path, 'wordnet_dict.py')).read())
        wordnet_id = wordnet_dict[class_index]
        
        self.images_path = os.path.join(images_path, 'train', wordnet_id)
        self.masks_path = os.path.join(masks_path, wordnet_id)
        
        image_names_file = os.path.join(self.masks_path, 'image_names_map.csv')
        image_names_df = pd.read_csv(image_names_file)
        
        image_names = []
        feature_indices_dict = defaultdict(list)
        for feature_index in feature_indices:
            image_names_feature = image_names_df[str(feature_index)].to_numpy()
            
            for i, image_name in enumerate(image_names_feature):
                image_names.append(image_name)                
                feature_indices_dict[image_name].append(feature_index)        
        
        self.image_names = np.unique(np.array(image_names))                
        self.feature_indices_dict = feature_indices_dict

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        curr_image_path = os.path.join(self.images_path, image_name + '.JPEG')

        image = Image.open(curr_image_path).convert("RGB")
        image_tensor = self.transform(image)
        
        feature_indices = self.feature_indices_dict[image_name]
        
        all_mask = np.zeros(image_tensor.shape[1:])
        for feature_index in feature_indices:            
            curr_mask_path = os.path.join(self.masks_path, 'feature_' + str(feature_index), image_name + '.JPEG')
            
            mask = np.asarray(Image.open(curr_mask_path))
            mask = (mask/255.)
            
            all_mask = np.maximum(all_mask, mask)

        all_mask = np.uint8(all_mask * 255)
        all_mask = Image.fromarray(all_mask)
        mask_tensor = self.transform(all_mask)
        return image_tensor, mask_tensor
    
class CompleteModel(ch.nn.Module):
    def __init__(self, model, CLIP_weights=None):
        super(CompleteModel, self).__init__()
        if CLIP_weights is None:
            self.mean = ch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
            self.std = ch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
            
            self.postprocess = lambda x: x
            self.model = model
        else:
            self.mean = ch.Tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).cuda()
            self.std = ch.Tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).cuda()
            
            self.last_weights = CLIP_weights
            self.postprocess = self.features_to_logits_map
            self.model = model.encode_image
            
        self.normalize = lambda x: (x - self.mean)/self.std
        
    def features_to_logits_map(self, features):
        features /= features.norm(dim=-1, keepdim=True)
        logits = 100. * (features @ self.last_weights)
        return logits
        
    def forward(self, inp):
        normalized_inp = self.normalize(inp)
        outputs = self.model(normalized_inp)
        logits = self.postprocess(outputs)
        return logits
    

def load_inspection_model(model_name):
    clip_models = ['clip_vit_b16', 'clip_vit_b32']
    timm_models = timm.list_models(pretrained=True)
    
    preprocess = None
    CLIP_weights = None
    if model_name in clip_models:
        if model_name == 'clip_vit_b16':
            model, preprocess = clip.load("ViT-B/16")
            weights_path = os.path.join('./models/clip_vit_b16_zeroshot_weights.npy')
        elif model_name == 'clip_vit_b32':
            model, preprocess = clip.load("ViT-B/32")
            weights_path = os.path.join('./models/clip_vit_b32_zeroshot_weights.npy')
            
        CLIP_weights = np.load(weights_path)
        CLIP_weights = ch.from_numpy(CLIP_weights)
        CLIP_weights = CLIP_weights.cuda()
    elif model_name in timm_models:
        model = timm.create_model(model_name, pretrained=True)
        model.eval()

    model.eval()
    model = CompleteModel(model, CLIP_weights=CLIP_weights)
    model = model.cuda()
    return model, preprocess

