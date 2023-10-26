import os
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from huggingface_hub import hf_hub_download
import subprocess
import pathlib
import logging


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.debug("Loaded dependencies successful!")


model_ckpt = "MCG-NJU/videomae-base"  # pre-trained model from which to fine-tune
batch_size = 8


# Loading the dataset
def downloadUCF101():
    hf_dataset_identifier = "sayakpaul/ucf101-subset"
    filename = "UCF101_subset.tar.gz"
    file_path = hf_hub_download(
        repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset"
    )

    if not os.path.exists(data_root):
        os.makedirs(data_root)

    subprocess.run(["tar", "xf", file_path, "-C", str(data_root)])


data_root = pathlib.Path("../data")
dataset_root_path = data_root / "UCF101_subset"
# if the dataset is not downloaded, download it
if not os.path.exists(dataset_root_path):
    downloadUCF101()

dataset_root_path = pathlib.Path(dataset_root_path)
video_count_train = len(list(dataset_root_path.glob("train/*/*.avi")))
video_count_val = len(list(dataset_root_path.glob("val/*/*.avi")))
video_count_test = len(list(dataset_root_path.glob("test/*/*.avi")))
video_total = video_count_train + video_count_val + video_count_test
print(f"Total videos: {video_total}")

all_video_file_paths = (
    list(dataset_root_path.glob("train/*/*.avi"))
    + list(dataset_root_path.glob("val/*/*.avi"))
    + list(dataset_root_path.glob("test/*/*.avi"))
)
print(all_video_file_paths[:5])

class_labels = sorted({str(path).split("/")[4] for path in all_video_file_paths})
label2id = {label: i for i, label in enumerate(class_labels)}
id2label = {i: label for label, i in label2id.items()}
print(f"Unique classes: {list(label2id.keys())}.")

# Loading the model
image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
model = VideoMAEForVideoClassification.from_pretrained(
    model_ckpt,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)
