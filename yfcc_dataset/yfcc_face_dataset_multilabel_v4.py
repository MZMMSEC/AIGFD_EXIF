import os
import pdb
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomCrop
from torchvision import transforms
import torch, cv2
import random
import json
import glob,pickle
from tqdm import tqdm

'''
这个dataset里包含augmented negative face samples
'''
def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(res):
    return Compose([
        # _convert_image_to_rgb,
        ToTensor(),
        # RandomCrop((res,res)),
        Resize((res,res)),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def _transform_1(res):
    return Compose([
        ToTensor(),
        transforms.RandomGrayscale(p=1),
        Resize((res, res)),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def _transform_2(res):
    return Compose([
        ToTensor(),
        transforms.RandomHorizontalFlip(),
        Resize((res, res)),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

from itertools import product
isos = ['low', 'medium', 'high', 'very high']
avs = ['ultra-wide', 'wide', 'medium', 'small']
ets = ['slow', 'medium', 'fast', 'ultra-fast']
fls = ['wide-angle', 'standard', 'medium-telephoto', 'telephoto']
combinations = list(product(isos, avs, ets, fls))
mapping = {combination: index for index, combination in enumerate(combinations)}

def bbox_crop_pil2cvt2pil(neg_imgpath, meta_data_dict, scale_factor=1.3):
    # pil loading
    input = Image.open(neg_imgpath).convert('RGB')
    # transform to cv2
    input = cv2.cvtColor(np.asarray(input),cv2.COLOR_RGB2BGR)
    # get the metadata based on path name
    if '_' in neg_imgpath: # first negative samples
        name_id = neg_imgpath.split('/')[-1].split('_')[0]
    else: # otherwise, positive samples
        name_id = neg_imgpath.split('/')[-1].split('.')[0]
    meta_data = meta_data_dict[name_id]

    left, top = round(meta_data['bounding_box'][0] * input.shape[0]), round(meta_data['bounding_box'][1] * input.shape[0])
    right, bottom = round(meta_data['bounding_box'][2] * input.shape[0]), round(meta_data['bounding_box'][3] * input.shape[0])
    size_bb = int(max(right - left, bottom - top) * scale_factor)
    center_x, center_y = (right + left) // 2, (bottom + top) // 2
    # 按scale扩大的bounding box的left top
    x1_ = max(int(center_x - size_bb // 2), 0)
    y1_ = max(int(center_y - size_bb // 2), 0)
    # 扩大的bounding box的size,但得考虑bounding box不能超出图像尺寸
    height, width = input.shape[0], input.shape[1]
    size_bb_x = min(width - x1_, size_bb)
    size_bb_y = min(height - y1_, size_bb)
    left, top = x1_, y1_

    cropped_face = input[top: top + size_bb_y, left: left + size_bb_x, :]
    cropped_pil_image = Image.fromarray(np.uint8(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)))
    return cropped_pil_image

def shuffle_lists(list1, list2):
    indices = list(range(len(list2)))
    random.shuffle(indices)
    shuffled_list2 = [list2[i] for i in indices]
    shuffled_list1 = [list1[i] for i in indices]
    return shuffled_list1, shuffled_list2

def random_select_index(lst, train_mode=True):
    # index = random.randint(0, len(lst) - 1)
    if train_mode:
        index = random.choice([0,0,0,0,1,2,3,4])
    else:
        index = 0
    selected_element = lst[index]
    return index, selected_element

class yfcc_face(Dataset):
    def __init__(self, face_root_path='./SSL_training_data/FDF/data/fdf/images/128',
                 exif_root_path="./SSL_training_data/fdf_ccby2_exif_update_filtered_v2/", # need to upload
                 photoid_vs_imgname='./SSL_training_data/FDF/data/id_vs_fdfName.pkl', # need to upload
                 neg_queue_root_path = './SSL_training_data/data/face_dataaug_neg', # need to upload
                 face_metainfo_root_path = './SSL_training_data/FDF/data/fdf/fdf_metainfo.json', # need to upload
                 resolution=224, face_scale=1.3, train_mode=True, cls_face_mode=False):
        super(yfcc_face, self).__init__()
        self.face_root_path = face_root_path
        self.exif_root_path = exif_root_path
        self.photoid_vs_imgname = photoid_vs_imgname
        self.transforms = _transform(resolution)
        self.neg_queue_root_path = neg_queue_root_path
        self.train_mode = train_mode
        self.res = resolution
        self.face_scale = face_scale
        self.cls_face_mode = cls_face_mode
        with open(face_metainfo_root_path) as file:
            self.metadata = json.load(file)

        self.imgs_q = []
        self.imgs_aug = []
        self.imgs = []

        # self.show_dist_exif()

        self.gather_possible_items(self.face_root_path, self.exif_root_path,
                                   self.photoid_vs_imgname, self.neg_queue_root_path)

        self.imgs_use = self.split_train_val(train_mode)

        print(f'total length of the dataset is {len(self.imgs_use)} faces...')

    def split_train_val(self, train=True):
        random.shuffle(self.imgs)

        if train:
            imgs_use = self.imgs[:-5000]
        else:
            imgs_use = self.imgs[-5000:]

        return imgs_use

    def __len__(self):
        return len(self.imgs_use)

    def __getitem__(self, idx):
        face_exif_items = self.imgs_use[idx]
        # get exif info and labels
        exif_num = face_exif_items['exif_num']
        exif_info_str = face_exif_items['exif_info_str']
        ISO = exif_num['ISO Speed Ratings']
        AV = exif_num['Aperture Value']
        FL = exif_num['Focal Length']
        ET = exif_num['Exposure Time']
        label = {}
        label['iso'] = ISO
        label['av'] = AV
        label['et'] = FL
        label['fl'] = ET

        # get img path
        imgpath_ls = face_exif_items['imgpath']
        img_idx, imgpath = random_select_index(imgpath_ls, train_mode=self.train_mode)
        # get img
        img = bbox_crop_pil2cvt2pil(imgpath, self.metadata, scale_factor=self.face_scale)
        img = self.transforms(img)
        if self.cls_face_mode:
            # get cls label for face
            if img_idx == 0:
                label['face'] = 0
            else:  # negaitive samples
                label['face'] = 1

        return img, label, exif_info_str


    def get_joint_label_mapping(self, label):
        comb = (isos[label['iso']], avs[label['av']], ets[label['et']], fls[label['fl']])
        joint_labels = mapping[comb]

        return joint_labels

    def get_multilabel(self, label):
        iso_labels = [0,0,0]
        av_labels = [0,0,0,0]
        et_labels = [0,0,0,0]
        fl_labels = [0,0,0,0]
        iso_labels[label['iso']] = 1
        av_labels[label['av']] = 1
        et_labels[label['et']] = 1
        fl_labels[label['fl']] = 1

        multilabel = []
        multilabel.extend(iso_labels)
        multilabel.extend(av_labels)
        multilabel.extend(et_labels)
        multilabel.extend(fl_labels)
        multilabel_tensor = torch.tensor(multilabel)
        return multilabel, multilabel_tensor

    def get_label_relative_4cls(self, value, input):
        if input <= value[0]:
            label = 0
        elif input > value[0] and input <= value[1]:
            label = 1
        elif input > value[1] and input <= value[2]:
            label = 2
        else:
            label = 3
        return label

    def get_label_relative(self, value, input):
        if input <= value[0]:
            label = 0
        elif input > value[0] and input <= value[1]:
            label = 1
        else:
            label = 2
        return label

    def gather_possible_items(self, face_root_path, exif_root_path, photoid_vs_imgname, neg_queue_root_path):
        with open(photoid_vs_imgname, 'rb') as file:
            photoid_vs_imgname_dict = pickle.load(file)

        face_paths = glob.glob(f"{face_root_path}/*.png")

        for face in tqdm(face_paths):
            imgname = face.split('/')[-1].split('.')[0]
            photo_id = photoid_vs_imgname_dict[imgname]
            exif_path = os.path.join(exif_root_path, photo_id+'.json')
            if not os.path.isfile(exif_path):
                continue

            exif_info, exif_info_str, exif_num = self.get_exif_str(exif_path)
            if exif_info is None:
                continue

            face_exif_items = {}
            face_path_list = []
            face_exif_items['exif_num'] = exif_num
            face_exif_items['exif_info_str'] = exif_info_str

            face_path_list.append(face)
            if self.train_mode:
                neg_queue_path_base = face.replace(face_root_path, neg_queue_root_path).split('.')[0]
                # pdb.set_trace()
                if not os.path.isfile(neg_queue_path_base + '_0.png'):
                    continue
                neg_queue_paths = []
                for num in range(4):
                    neg_queue_paths.append(
                        neg_queue_path_base + '_' + str(num) + '.png'
                    )


                face_path_list.extend(neg_queue_paths)

            face_exif_items['imgpath'] = face_path_list
                # face_exif_items['neg_queue_path'] = neg_queue_paths

            self.imgs.append(face_exif_items)

    def get_exif_str(self, exif_path):
        with open(exif_path, 'r') as file:
            exif = json.load(file)

        exif_info = {}
        exif_info['ISO Speed Ratings'] = exif['EXIF']['ISO Speed Ratings']
        exif_info['Aperture Value'] = exif['EXIF']['Aperture Value']
        exif_info['Exposure Time'] = exif['EXIF']['Exposure Time']
        exif_info['Focal Length'] = exif['EXIF']['Focal Length']

        if exif_info['Aperture Value'] == 'F':
            return None, None, None

        exif_num = self.get_exif_statistics(exif_info)

        exif_str = ", ".join([f"{key}: {value}" for key, value in exif_info.items()])

        return exif_info, exif_str, exif_num

    def get_exif_statistics(self, exif_info):
        try:
            ISO = float(exif_info['ISO Speed Ratings'])
            AV = float(exif_info['Aperture Value'].split('F')[-1])
            FL = float(exif_info['Focal Length'].split(' mm')[0])
            ET = float(exif_info['Exposure Time'].split('sec')[0].split('1/')[-1])
        except:
            ISO = float(exif_info['ISO Speed Ratings'])
            AV = float(eval(exif_info['Aperture Value'].split('F')[-1]))
            FL = float(exif_info['Focal Length'].split(' mm')[0])
            ET = float(eval(exif_info['Exposure Time'].split('sec')[0]))

        exif_num = {}
        exif_num['ISO Speed Ratings'] = ISO
        exif_num['Aperture Value'] = AV
        exif_num['Focal Length'] = FL
        exif_num['Exposure Time'] = ET
        return exif_num

    def show_dist_exif(self):
        ISO_ls = []
        AV_ls = []
        ET_ls = []
        FL_ls = []

        with open(self.photoid_vs_imgname, 'rb') as file:
            photoid_vs_imgname_dict = pickle.load(file)

        face_paths = glob.glob(f"{self.face_root_path}/*.png")

        for face in tqdm(face_paths):
            imgname = face.split('/')[-1].split('.')[0]
            photo_id = photoid_vs_imgname_dict[imgname]
            exif_path = os.path.join(self.exif_root_path, photo_id+'.json')
            if not os.path.isfile(exif_path):
                # pdb.set_trace()
                continue

            # pdb.set_trace()
            exif_info, _, _ = self.get_exif_str(exif_path)
            if exif_info is None:
                continue

            try:
                ISO = float(exif_info['ISO Speed Ratings'])
                AV = float(exif_info['Aperture Value'].split('F')[-1])
                FL = float(exif_info['Focal Length'].split(' mm')[0])
                ET = float(exif_info['Exposure Time'].split('sec')[0].split('1/')[-1])
            except:
                ISO = float(exif_info['ISO Speed Ratings'])
                try:
                    AV = float(eval(exif_info['Aperture Value'].split('F')[-1]))
                except:
                    continue
                FL = float(exif_info['Focal Length'].split(' mm')[0])
                ET = float(eval(exif_info['Exposure Time'].split('sec')[0]))

            ISO_ls.append(ISO)
            AV_ls.append(AV)
            ET_ls.append(ET)
            FL_ls.append(FL)

        ISO_4cls = split_list_by_size(ISO_ls) # 200, 400, 800
        AV_4cls = split_list_by_size(AV_ls) # F2.9, F4.5, F5.7
        ET_4cls = split_list_by_size(ET_ls) # 1/60, 1/200, 1/500,
        FL_4cls = split_list_by_size(FL_ls) # 20, 50, 125, 922
        pdb.set_trace()

def split_list_by_size(lst):
  """将列表按照元素大小大致均分成4份。

  Args:
    lst: 要分割的列表。

  Returns:
    一个包含4个子列表的列表。
  """
  lst.sort()
  n = len(lst)
  quarter = math.ceil(n / 4)
  return [lst[i:i+quarter] for i in range(0, n, quarter)]

def collect_possible_face_for_text():
    dataset = yfcc_face(train_mode=True)

    imgname_ls = []
    for data in tqdm(dataset.imgs_use):
        img_name = data['imgpath'].split('/')[-1]
        imgname_ls.append(img_name)

    print(len(imgname_ls))

    with open('train_data.pkl', 'wb') as file:
        pickle.dump(imgname_ls, file)


import matplotlib.pyplot as plt
import math, pickle
if __name__ == '__main__':
    train_dataset = yfcc_face(train_mode=True, cls_face_mode=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    for batch in train_dataloader:
        img, label, exif_info_str = batch
        pdb.set_trace()
