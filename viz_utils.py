import os
import numpy as np
import cv2
from PIL import Image

import torch as ch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

from robustness import model_utils 
from robustness import datasets as dataset_utils

class ImageNetSubset(Dataset):
    def __init__(self, root, image_indices):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()])

        self.imagenet = datasets.ImageNet(root = root, split = 'train') 
        self.image_indices = image_indices

    def __len__(self):
        return len(self.image_indices)

    def __getitem__(self, index):
        image_index = self.image_indices[index]
        image, _ = self.imagenet.__getitem__(image_index)
        image = self.transform(image)
        return image
    
def load_robust_model(imagenet_path):
    dataset_function = getattr(dataset_utils, 'ImageNet')
    dataset = dataset_function(imagenet_path)

    model_kwargs = {
        'arch': 'resnet50',
        'dataset': dataset,
        'resume_path': f'./models/robust_resnet50.pth',
        'parallel': False
    }
    model, _ = model_utils.make_and_restore_model(**model_kwargs)
    model.eval()
    return model

def generate_features(model, dataset, batch_size=128):
    data_loader = DataLoader(dataset, batch_size=batch_size)
    features_all = []
    
    total = 0
    for images in data_loader:
        images = images.cuda()
        
        (_, features), _ = model(images, with_latent=True)
        features = features.detach()
        total += len(images)
        
        features_all.append(features.detach().cpu().numpy())
        
    features_all = np.concatenate(features_all, axis=0)
    return features_all

def load_images(indices, dataset):
    img_list = []
    for idx in indices:
        img = dataset.__getitem__(idx)
        img_list.append(img)
    img_tensor = ch.stack(img_list, dim=0)
    return img_tensor

def load_images_fnames(fnames, dataset_path):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()])

    img_list = []
    for fname in fnames:
        img = Image.open(os.path.join(dataset_path, fname))
        img = transform(img)
        img_list.append(img)
        
    img_tensor = ch.stack(img_list, dim=0)
    return img_tensor



def compute_feature_maps(images, model, layer_name='layer4'):
    images = images.cuda()
    normalizer_module = model._modules['normalizer']
    feature_module = model._modules['model']
    x = normalizer_module(images)
    for name, module in feature_module._modules.items():
        x = module(x)
        if name == layer_name:
            break
    return x

def compute_nams(model, images, feature_index, layer_name='layer4'):
    b_size = images.shape[0]
    feature_maps = compute_feature_maps(images, model, layer_name=layer_name)
    nams = (feature_maps[:, feature_index, :, :]).detach()
    nams_flat = nams.view(b_size, -1) 
    nams_max, _ = ch.max(nams_flat, dim=1, keepdim=True)
    nams_flat = nams_flat/nams_max
    nams = nams_flat.view_as(nams)

    nams_resized = []
    for nam in nams:
        nam = nam.cpu().numpy()
        nam = cv2.resize(nam, images.shape[2:])
        nams_resized.append(nam)
    nams = np.stack(nams_resized, axis=0)
    nams = ch.from_numpy(1-nams)
    return nams

def compute_heatmaps(imgs, masks):
    heatmaps = []
    for (img, mask) in zip(imgs, masks):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap + np.float32(img)
        heatmap = heatmap / np.max(heatmap)
        heatmaps.append(heatmap)
    heatmaps = np.stack(heatmaps, axis=0)
    heatmaps = ch.from_numpy(heatmaps).permute(0, 3, 1, 2)
    return heatmaps

def grad_step(adv_inputs, grad, step_size):
    l = len(adv_inputs.shape) - 1
    grad_norm = ch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, *([1]*l))
    scaled_grad = grad / (grad_norm + 1e-10)
    return adv_inputs + scaled_grad * step_size

def feature_attack(model, seed_images, feature_indices, eps=500, 
                   step_size=10, iterations=100):
    seed_images = seed_images.cuda()
    batch_size = seed_images.shape[0]
    for i in range(iterations+1):
        seed_images.requires_grad_()

        (_, features), _ = model(seed_images, with_latent=True)
        features_select = features[ch.arange(batch_size), feature_indices]
        
        if i==iterations:
            seed_images = seed_images.detach().cpu()
            features_select = features_select.detach().cpu().numpy()
            break
            
        adv_loss = features_select.sum()
        grads = ch.autograd.grad(adv_loss, [seed_images])[0]

        seed_images = grad_step(seed_images.detach(), grads, step_size)
        seed_images = ch.clamp(seed_images, min=0., max=1.)
    return seed_images, features_select

def create_images(indices_high, feature_index, dataset, robust_model):
    images_highest = load_images(indices_high, dataset)
    images_nams = compute_nams(robust_model, images_highest, feature_index, layer_name='layer4')
    
    images_heatmaps = compute_heatmaps(images_highest.permute(0, 2, 3, 1), images_nams)
    images_attacks, _ = feature_attack(robust_model, images_highest, feature_index)
    return images_highest, images_heatmaps, images_attacks
