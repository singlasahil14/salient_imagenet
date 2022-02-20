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
    def __init__(self, main_dir, add_noise_mean=0., batch_size=32):
        self.main_dir = main_dir
        self.add_noise_mean = add_noise_mean
        self.batch_size = batch_size

    def add_gaussian_noise(self, images, soft_masks, add_noise_std):
        gaussian_noise = self.add_noise_mean + add_noise_std * ch.randn(*images.shape)
        
        images_n = images + (gaussian_noise * soft_masks)
        images_n = ch.clamp(images_n, 0., 1.)
        return images_n
        
    def corrupted_images(self, images, masks, corruption_type="add", add_noise_std=None):
        assert corruption_type in ["none", "add"]
        
        images = images.float().cpu()
        masks = masks.float().cpu()

        if corruption_type == "add":
            images_corrupted = self.add_gaussian_noise(images, masks, add_noise_std)
        elif corruption_type == "none":
            images_corrupted = images
        return images_corrupted
    
    def compute_noisy_acc(self, inspection_model, class_index, feature_indices, 
                          add_noise_std, preprocess=None):
        preds_all = []
        preds_noisy_all = []
        
        salient_imagenet = SalientImageNet(self.main_dir, class_index, feature_indices, 
                                            preprocess=preprocess)
        total = len(salient_imagenet)
        
        data_loader = DataLoader(salient_imagenet, batch_size=self.batch_size, shuffle=False)
        
        for images_batch, soft_masks_batch in data_loader:
            images_batch = images_batch.float().cuda()            
            with ch.no_grad():
                logits_batch = inspection_model(images_batch)
            preds_batch = ch.argmax(logits_batch, dim=1).cpu().numpy()            
            preds_all.append(preds_batch)

            
            images_noisy_batch = self.corrupted_images(images_batch, soft_masks_batch, 
                                                       corruption_type="add", 
                                                       add_noise_std=add_noise_std)
            images_noisy_batch = images_noisy_batch.float().cuda()
            with ch.no_grad():
                logits_noisy_batch = inspection_model(images_noisy_batch)
            preds_noisy_batch = ch.argmax(logits_noisy_batch, dim=1).cpu().numpy()
            preds_noisy_all.append(preds_noisy_batch)

        preds_all = np.concatenate(preds_all, axis=0)
        preds_noisy_all = np.concatenate(preds_noisy_all, axis=0)
        
        clean_acc = np.sum(preds_all == class_index)/total
        noisy_acc = np.sum(preds_noisy_all == class_index)/total

        return clean_acc, noisy_acc
    
    
class SalientImageNet(Dataset):
    def __init__(self, main_dir, class_index, feature_indices, 
                 resize_size=224, preprocess=None):
        if preprocess is None:
            self.transform = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.CenterCrop(resize_size),
                transforms.ToTensor()
            ])
        else:
            self.transform = preprocess

        wordnet_dict = eval(open(os.path.join(main_dir, 'wordnet_dict.py')).read())
        wordnet_id = wordnet_dict[class_index]
        
        class_path = os.path.join(main_dir, wordnet_id)

        image_fnames_file = os.path.join(class_path, 'image_fnames.csv')
        image_fnames_df = pd.read_csv(image_fnames_file)
        
        feature_indices_dict = defaultdict(list)
        image_paths = []
        image_fnames = []
        for feature_index in feature_indices:
            feature_path = os.path.join(class_path, 'feature_' + str(feature_index))
            images_path = os.path.join(feature_path, 'images')            
            
            image_fnames_feature = image_fnames_df[str(feature_index)].to_numpy()
            
            for i, image_fname in enumerate(image_fnames_feature):
                image_fnames.append(image_fname)
                
                image_path = os.path.join(images_path, str(image_fname) + '.jpeg')
                image_paths.append(image_path)
                
                feature_indices_dict[image_fname].append(feature_index)        
        
        image_fnames = np.array(image_fnames)
        self.image_fnames, unique_indices = np.unique(image_fnames, return_index=True)        
                
        image_paths = np.array(image_paths)
        self.image_paths = image_paths[unique_indices]
        
        self.feature_indices_dict = feature_indices_dict
        
        self.class_path = class_path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_fname = self.image_fnames[index]
        image_path = self.image_paths[index]

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        
        feature_indices = self.feature_indices_dict[image_fname]
        
        all_mask = np.zeros(image.size)
        for feature_index in feature_indices:
            feature_path = os.path.join(self.class_path, 'feature_' + str(feature_index))
            cams_path = os.path.join(feature_path, 'cams')

            cam_path = os.path.join(cams_path, str(image_fname) + '.jpeg')
            mask = np.asarray(Image.open(cam_path))
            mask = (mask/255.)
            
            all_mask = np.maximum(all_mask, mask)
        
        
        all_mask = np.uint8(all_mask * 255)
        all_mask = Image.fromarray(all_mask)
        mask_tensor = self.transform(all_mask)
        return image_tensor, mask_tensor
    
class CompleteModel(ch.nn.Module):
    def __init__(self, model, last_weights=None):
        super(CompleteModel, self).__init__()
        if last_weights is None:
            self.mean = ch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
            self.std = ch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
            self.normalize = lambda x: (x - self.mean)/self.std
            
            self.postprocess = lambda x: x
            self.model = model
        else:
            self.normalize = lambda x: x
            
            self.last_weights = last_weights
            self.postprocess = self.features_to_logits_map
            self.model = model.encode_image
        
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
    last_weights = None
    if model_name in clip_models:
        if model_name == 'clip_vit_b16':
            model, preprocess = clip.load("ViT-B/16")
            weights_path = os.path.join('./models/clip_vit_b16_zeroshot_weights.npy')
        elif model_name == 'clip_vit_b32':
            model, preprocess = clip.load("ViT-B/32")
            weights_path = os.path.join('./models/clip_vit_b32_zeroshot_weights.npy')
            
        linear_weights = np.load(weights_path)
        linear_weights = ch.from_numpy(linear_weights)
        last_weights = linear_weights.cuda()
    elif model_name in timm_models:
        model = timm.create_model(model_name, pretrained=True)
        model.eval()

    model.eval()
    model = CompleteModel(model, last_weights=last_weights)
    model = model.cuda()
    return model, preprocess

