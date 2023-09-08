import logging
import pandas as pd
import fastdup
import plotly.express as px
from sklearn.manifold import TSNE

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def plot_tsne_3d(feature_vec, connected_components_df, filenames, output_path):
    """
    Function to plot 3D t-SNE scatter plot and save it html file.
    """
    # Instantiate and fit TSNE model
    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
    tsne_result = tsne.fit_transform(feature_vec)
    component_id = connected_components_df["component_id"].to_numpy()
    # Create a pandas dataframe with the embeddings and labels
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

    # fig.write_image("work_dir/embeddings_tsne.png")
    fig.write_html("work_dir/embeddings_tsne2.html")


if __name__ == "__main__":
    df_embeddings = pd.read_csv("work_dir/vec_df.csv")
    filenames = df_embeddings["Image_Name"]
    feature_vec = df_embeddings.drop("Image_Name", axis=1)
    output_path = "work_dir/embeddings_tsne3.html"

    fd = fastdup.create(input_dir="../Oxford Pets Dataset/")
    fd.run(run_cc=True)
    connected_components_df, _ = fd.connected_components()
    print(connected_components_df["component_id"].to_numpy().shape)
    print(filenames.shape)
    plot_tsne_3d(feature_vec, connected_components_df[:130], filenames, output_path)
