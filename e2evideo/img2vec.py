import os
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.debug("Loaded dependencies successful!")

model = models.resnet18(pretrained=True)
layer = model._modules.get("avgpool")
model.eval()

logger.debug("loaded model successful!")

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()


def get_vector(image_name):
    img = Image.open(image_name)
    t_img = normalize(to_tensor(scaler(img))).unsqueeze(0)
    my_embedding = torch.zeros(512)

    def copy_data(m, i, o):
        my_embedding.copy_(o.data.squeeze())

    h = layer.register_forward_hook(copy_data)
    model(t_img)
    h.remove()
    return my_embedding


vec_list = []
dir_path = "../Oxford Pets Dataset/"


for img_name in tqdm(os.listdir(dir_path)[:130]):
    img_path = os.path.join(dir_path, img_name)
    vec = get_vector(img_path)
    vec = vec.numpy().tolist()
    vec_dict = {"Image_Name": img_name}
    for i, v in enumerate(vec):
        vec_dict[f"Vector_{i}"] = v
    vec_list.append(vec_dict)

vec_df = pd.DataFrame(vec_list)
vec_df.to_csv("work_dir/vec_df.csv", index=False)
