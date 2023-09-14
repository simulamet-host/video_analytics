import os
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import fastdup
from sklearn.manifold import TSNE
import plotly.express as px
import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.debug("Loaded dependencies successful!")


class FeatureExtractor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.to_tensor = transforms.ToTensor()

    def get_vector(self, image_name, layer, model):
        img = Image.open(image_name)
        t_img = self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0)
        my_embedding = torch.zeros(512)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data.squeeze())

        h = layer.register_forward_hook(copy_data)
        model(t_img)
        h.remove()
        return my_embedding

    def extract_img_vector(self):
        model = models.resnet18(pretrained=True)
        layer = model._modules.get("avgpool")
        model.eval()
        logger.debug("{model} loaded successful!")

        vec_list = []

        for img_name in tqdm(os.listdir(self.input_path)[:130]):
            img_path = os.path.join(self.input_path, img_name, layer, model)
            vec = self.get_vector(img_path)
            vec = vec.numpy().tolist()
            vec_dict = {"Image_Name": img_name}
            for i, v in enumerate(vec):
                vec_dict[f"Vector_{i}"] = v
            vec_list.append(vec_dict)

        vec_df = pd.DataFrame(vec_list)
        vec_df.to_csv("{self.output_path}/vec_df.csv", index=False)
        return vec_df

    def extract_dinov2_features(self):
        fd = fastdup.create(input_dir=self.input_path)
        fd.run(model_path="dinov2s", cc_threshold=0.8)

        filenames, feature_vec = fastdup.load_binary_feature(
            f"{self.output_path}/atrain_features.dat", d=384
        )
        logger.info("Embedding dimensions %s", feature_vec.shape)
        return filenames, feature_vec


def plot_tsne_3d(feature_vec, connected_components_df, filenames, output_path):
    """
    Function to plot 3D t-SNE scatter plot and save it html file.
    """
    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
    tsne_result = tsne.fit_transform(feature_vec)
    component_id = connected_components_df["component_id"].to_numpy()
    df = pd.DataFrame(
        {
            "tsne_1": tsne_result[:, 0],
            "tsne_2": tsne_result[:, 1],
            "tsne_3": tsne_result[:, 2],
            "component": component_id,
            "filename": filenames,
        }
    )

    fig = px.scatter_3d(
        df,
        x="tsne_1",
        y="tsne_2",
        z="tsne_3",
        color="component",
        opacity=0.5,
        hover_data=["component", "filename"],
    )

    fig.write_html("work_dir/embeddings_tsne2.html")


if __name__ == "__main__":
    input_path = "../data/Oxford Pets Dataset/"
    output_path = "./work_dir"
    fe = FeatureExtractor(input_path, output_path)
    filenames, feature_vec = fe.extract_dinov2_features()

    connected_components_df = pd.read_csv(
        os.path.join("work_dir", "connected_components.csv")
    )
    plot_tsne_3d(
        feature_vec,
        connected_components_df,
        filenames,
        "./work_dir/embeddings_dinvo2.html",
    )
