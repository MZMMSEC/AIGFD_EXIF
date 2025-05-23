import pdb

import torch
import copy
import clip
import numpy as np
# from transformers import DistilBertModel, DistilBertConfig
# from transformers import AlbertModel, AlbertConfig
# from transformers import RobertaModel,RobertaConfig
from torchvision.models import resnet50
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def init_model_classification(model_name="ViT-B/32", device=None, state_dict_path=None, split_gpus=False,
                              visual="RN50", pretrained=True,
                              face_embeds=False, ranking=False, exif_only=False):

    clipNet, _ = clip.load(model_name,device=device,jit=False)
    if device == "cpu":
          clipNet.float()
    else :
        clip.model.convert_weights(clipNet)
    logit_scale = clipNet.logit_scale.exp()
                                
    if ranking and not exif_only:
        clipNet = EXIFNet_classfication_5heads_v2(clipNet, visual=visual)
    elif exif_only:
        clipNet = EXIFNet_classfication_5heads_v3(clipNet, visual=visual)
    else:
        clipNet = EXIFNet_classfication_FaceOnly(clipNet, visual=visual)

    if (state_dict_path and not face_embeds) or (state_dict_path and pretrained):
        if device == 'cpu':
            checkpoint = torch.load(state_dict_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(state_dict_path)
        msg = clipNet.load_state_dict(checkpoint['model'], strict=False)
        print(f"load pretraining model {state_dict_path}")
        print(msg)

    if split_gpus:
        print("Multiple GPU setting!")
        if torch.cuda.device_count() > 1:
            print("let's use", torch.cuda.device_count(), "GPUs!")
        clipNet = torch.nn.DataParallel(clipNet)
    clipNet.to(device)

    return clipNet, logit_scale.detach()

class EXIFNet_classfication_5heads_v2(torch.nn.Module):
    def __init__(self, model, visual="RN50"):
        super().__init__()
        self.model = model
        del self.model.visual
        del self.model.transformer
        del self.model.text_projection
        if visual == "RN50":
            image_encoder = resnet50(pretrained=True)
        else:
            raise NotImplementedError
        image_encoder.fc = torch.nn.Linear(2048, 768)
        self.model.visual = image_encoder

        self.fc_face = nn.Linear(768, 1)

        self.fc_iso = nn.Linear(768, 1)
        self.fc_av = nn.Linear(768, 1)
        self.fc_et = nn.Linear(768, 1)
        self.fc_fl = nn.Linear(768, 1)

    def encode_image(self, image):
        return self.model.visual(image)

    def encoder_cls(self, image_embeds):
        logits_iso = self.fc_iso(image_embeds)
        logits_av = self.fc_av(image_embeds)
        logits_et = self.fc_et(image_embeds)
        logits_fl = self.fc_fl(image_embeds)
        logits_face = self.fc_face(image_embeds)
        return logits_iso, logits_av, logits_et, logits_fl, logits_face

    def forward(self, image):
        image = image.to(DEVICE)
        image_embeds = self.encode_image(image)
        logits_iso, logits_av, logits_et, logits_fl, logits_face = self.encoder_cls(image_embeds)

        return logits_iso, logits_av, logits_et, logits_fl, logits_face

class EXIFNet_classfication_5heads_v3(torch.nn.Module):
    def __init__(self, model, visual="RN50"):
        super().__init__()
        self.model = model
        del self.model.visual
        del self.model.transformer
        del self.model.text_projection
        if visual == "RN50":
            image_encoder = resnet50(pretrained=True)
        else:
            raise NotImplementedError
        image_encoder.fc = torch.nn.Linear(2048, 768)
        self.model.visual = image_encoder

        self.fc_iso = nn.Linear(768, 1)
        self.fc_av = nn.Linear(768, 1)
        self.fc_et = nn.Linear(768, 1)
        self.fc_fl = nn.Linear(768, 1)

    def encode_image(self, image):
        return self.model.visual(image)

    def encoder_cls(self, image_embeds):
        logits_iso = self.fc_iso(image_embeds)
        logits_av = self.fc_av(image_embeds)
        logits_et = self.fc_et(image_embeds)
        logits_fl = self.fc_fl(image_embeds)
        return logits_iso, logits_av, logits_et, logits_fl

    def forward(self, image):
        image = image.to(DEVICE)
        image_embeds = self.encode_image(image)
        logits_iso, logits_av, logits_et, logits_fl = self.encoder_cls(image_embeds)

        return logits_iso, logits_av, logits_et, logits_fl

class EXIFNet_classfication_FaceOnly(torch.nn.Module):
    def __init__(self, model, visual="RN50"):
        super().__init__()
        self.model = model
        del self.model.visual
        del self.model.transformer
        del self.model.text_projection
        if visual == "RN50":
            image_encoder = resnet50(pretrained=True)
        else:
            raise NotImplementedError
        image_encoder.fc = torch.nn.Linear(2048, 768)
        self.model.visual = image_encoder

        self.fc_joint = nn.Linear(768, 2)


    def encode_image(self, image):
        return self.model.visual(image)

    def forward(self, image):
        image = image.to(DEVICE)
        image_embeds = self.encode_image(image)

        logits_joint = self.fc_joint(image_embeds)

        return logits_joint
