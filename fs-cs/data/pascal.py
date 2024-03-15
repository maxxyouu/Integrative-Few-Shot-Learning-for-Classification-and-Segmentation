r""" PASCAL-5i few-shot semantic segmentation dataset code from NOTE: L-seg """
import os

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from common import fda_utils as fda
from torchvision import transforms


class DatasetPASCAL(Dataset):
    def __init__(self, datapath, fold, transform, split, way, shot, fda, use_original_imgsize=True):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 20
        self.way = way # NOTE: in our case, by default it is always one way
        self.benchmark = 'pascal'
        self.shot = shot
        self.fda = fda
        self.use_original_imgsize = use_original_imgsize

        self.img_path = os.path.join(datapath, 'VOC2012/JPEGImages/')
        self.ann_path = os.path.join(datapath, 'VOC2012/SegmentationClassAug/')
        self.transform = transform

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]
        self.normalization = transforms.Normalize(self.img_mean, self.img_std)

        # from the implementation, https://github.com/YanchaoYang/FDA , do the fda on the original image, then do the augmentation (best guess).

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else min(1000, len(self.img_metadata))

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = self.load_frame(query_name, support_names) # cmask stands for class mask, load the actual images from the image names

        query_class_presence = [s_c in torch.unique(query_cmask) for s_c in [class_sample+1]]  # needed - 1

        query_img = self.transform(query_img)
        # if not self.use_original_imgsize:
        if self.split == 'trn':
            query_cmask = F.interpolate(query_cmask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask.float(), class_sample)
        
        if self.shot:
            # keep all the support images into one tensor
            support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

            support_masks = []
            support_ignore_idxs = [] # these are the border pixels
            for scmask in support_cmasks:
                scmask = F.interpolate(scmask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
                support_mask, support_ignore_idx = self.extract_ignore_idx(scmask, class_sample)
                support_masks.append(support_mask)
                support_ignore_idxs.append(support_ignore_idx)
            
            # keep all the support masks as one tensors
            support_masks = torch.stack(support_masks)
            support_ignore_idxs = torch.stack(support_ignore_idxs)
        else:
            support_masks = []
            support_ignore_idxs = []

        # NOTE: perform fda operation from each episodes
        if self.fda > 0.:
            std_query, std_supports = self.episode_style_standardization(query_img, support_imgs)
            query_img, support_imgs = std_query, std_supports
        
        # use this batch information for testing
        batch = {'query_img': query_img,
                'query_mask': query_mask,
                'query_name': query_name,
                'query_ignore_idx': query_ignore_idx,

                'org_query_imsize': org_qry_imsize,

                'support_imgs': support_imgs,
                'support_masks': support_masks,
                'support_names': support_names,
                'support_ignore_idxs': support_ignore_idxs,

                'support_classes': torch.tensor([class_sample+1]), # class_sample + 1 is the class id (image name label) not class index id
                'query_class_presence': torch.tensor(query_class_presence)}

        return batch

    def episode_style_standardization(self, query_img, support_imgs):
        """ use the query as style anchor and standardize the style of support
        check implementation here: https://stackoverflow.com/questions/71515439/equivalent-to-torch-rfft-in-newest-pytorch-version
        Args:
            query_img (tensor): of shape [c, h, w]
            support_imgs (tesnor): of shape [s, c, h, w]
        """
        # style standardization and then mean subtraction and normalization 
        spprts_in_trg = []
        qry_img_cpy = query_img.clone()
        support_imgs_cpy = support_imgs.clone()
        for k in range(support_imgs_cpy.shape[0]):
            spprt = support_imgs_cpy[k, :] # [c, h, w]

            # convert to np then convert back to tensor type.
            temp = fda.FDA_source_to_target_np(spprt.detach().cpu().numpy(), qry_img_cpy.detach().cpu().numpy(), L=self.fda)
            temp = torch.from_numpy(temp).to(spprt.device)

            # for pytorch implementation, check https://arxiv.org/pdf/2303.06088.pdf
            # temp2 = fda.FDA_source_to_target(spprt.unsqueeze(0), qry_img_cpy.unsqueeze(0)).squeeze(0)
            # torch.testing.assert_close(temp, temp2, check_stride=False)
            spprts_in_trg.append(temp)

        spprts_in_trg = torch.stack(spprts_in_trg, dim=0).float()

        # after style standardization, normalize the images in the episode.
        std_query = self.normalization(query_img)
        std_supports = self.normalization(spprts_in_trg)
        return std_query, std_supports

    def extract_ignore_idx(self, mask, class_id):
        # only get the class of interest here
        boundary = (mask / 255).floor()
        mask[mask != class_id + 1] = 0
        mask[mask == class_id + 1] = 1

        return mask, boundary

    def load_frame(self, query_name, support_names):
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_imgs = [self.read_img(name) for name in support_names]
        support_masks = [self.read_mask(name) for name in support_names]

        org_qry_imsize = query_img.size

        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '.png')))
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')

    def sample_episode(self, idx):
        # NOTE: idx is not a class id
        # return a triple of the query image name (1 image), support image names (length of self.shot), class (one label)

        # recall img_metadata is a list of tuples
        query_name, class_sample = self.img_metadata[idx]

        support_names = []
        if self.shot:
            while True:  # keep sampling support set if query == support

                # sample the image that contains the corresponding query class with no replacement
                support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
                if query_name != support_name: support_names.append(support_name)
                if len(support_names) == self.shot: break

        # NOTE: note that class_sample is already start from 1 derived from the image name
        return query_name, support_names, class_sample

    def build_class_ids(self):
        # each fold has different number of classes
        nclass_trn = self.nclass // self.nfolds

        # train fold i has the same set of classes as val fold i
        # note that the class id starts from 1 from the image name => that is why you minus 1 in build_img_metadata function
        # because the class id here are the index start from 0
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        
        # return the fold of interest only.
        if self.split == 'trn':
            return class_ids_trn
        else:
            return class_ids_val

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            fold_n_metadata = os.path.join(f'fs-cs/data/splits/pascal/{split}/fold{fold_id}.txt')
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]

            # note that data.split('__')[0] is the image name and int(data.split('__')[1]) is the class id but minus 1 here
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'trn':  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:  # Skip target fold, the rest of the three folds are used for training
                    continue
                img_metadata += read_metadata(self.split, fold_id)
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)

        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))

        # this return a list of tuples of (image id, class id)
        return img_metadata

    def build_img_metadata_classwise(self):
        # collect all images of the same class into its dictionary key
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        # collect per-class training samples with the target classes be empty.
        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        return img_metadata_classwise
