import fastdup
import logging
import os
import pandas as pd

import plotly.express as px
from sklearn.manifold import TSNE


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

fd = fastdup.create(input_dir="../Oxford Pets Dataset/")
fd.run(model_path="dinov2s", cc_threshold=0.8)
# fd.vis.component_gallery(save_path="dinov2s")

filenames, feature_vec = fastdup.load_binary_feature(
    "work_dir/atrain_features.dat", d=384
)
print("Embedding dimensions", feature_vec.shape)


connected_components_df = pd.read_csv(
    os.path.join("work_dir", "connected_components.csv")
)
component_id = connected_components_df["component_id"].to_numpy()


X = feature_vec
y = component_id

# Instantiate and fit TSNE model
tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
tsne_result = tsne.fit_transform(X)

# Create a pandas dataframe with the embeddings and labels
df = pd.DataFrame(
    {
        "tsne_1": tsne_result[:, 0],
        "tsne_2": tsne_result[:, 1],
        "tsne_3": tsne_result[:, 2],
        "component": y,
        "filename": filenames,
    }
)

# Create a Plotly 3D scatter plot with colored points
fig = px.scatter_3d(
    df,
    x="tsne_1",
    y="tsne_2",
    z="tsne_3",
    color="component",
    opacity=0.5,
    hover_data=["component", "filename"],
)

# fig.write_image("work_dir/embeddings_tsne.png")
fig.write_html("work_dir/embeddings_tsne.html")
