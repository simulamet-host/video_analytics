import fastdup
import logging
import os
import pandas as pd

from e2evideo import embedding_vis

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

fd = fastdup.create(input_dir="../Oxford Pets Dataset/")
fd.run(model_path="dinov2s", cc_threshold=0.8)

filenames, feature_vec = fastdup.load_binary_feature(
    "work_dir/atrain_features.dat", d=384
)
logger.info("Embedding dimensions", feature_vec.shape)


connected_components_df = pd.read_csv(
    os.path.join("work_dir", "connected_components.csv")
)
component_id = connected_components_df["component_id"].to_numpy()

embedding_vis.plot_tsne_3d(
    feature_vec, component_id, filenames, "work_dir/embeddings_dinvo2.html"
)
