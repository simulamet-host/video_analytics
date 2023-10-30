import os
import argparse
import numpy as np
import torch

from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
    TrainingArguments,
    Trainer,
)

from huggingface_hub import hf_hub_download
import subprocess
import pathlib
import logging

import pytorchvideo.data
import evaluate

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.debug("Loaded dependencies successful!")


# Loading the dataset
def downloadUCF101():
    hf_dataset_identifier = "sayakpaul/ucf101-subset"
    filename = "UCF101_subset.tar.gz"
    file_path = hf_hub_download(
        repo_id=hf_dataset_identifier,
        filename=filename,
        repo_type="dataset",
        use_auth_token=False,
    )

    if not os.path.exists(data_root):
        os.makedirs(data_root)

    subprocess.run(["tar", "xf", file_path, "-C", str(data_root)])


def investigate_video(sample_video):
    """Utility to investigate the keys present in a single video sample."""
    print("\n Keys in the video sample:")
    for k in sample_video:
        if k == "video":
            print(k, sample_video["video"].shape)
        else:
            print(k, sample_video[k])

    print(f"Video label: {id2label[sample_video[k]]} \n")


def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions."""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def collate_fn(examples):
    """The collation function to be used by `Trainer` to prepare data batches."""
    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def train_the_model():
    train_results = trainer.train()
    logger.info("Finished training.")
    print(train_results)
    print(trainer.evaluate(test_dataset))
    logger.info("Finished evaluating.")

    trainer.save_model()
    test_results = trainer.evaluate(test_dataset)
    trainer.log_metrics("test", test_results)
    trainer.save_metrics("test", test_results)
    trainer.save_state()
    logger.info("Finished saving model and results.")


def run_inference(model, video):
    """Utility to run inference given a model and test video.

    The video is assumed to be preprocessed already.
    """
    # (num_frames, num_channels, height, width)
    perumuted_sample_test_video = video.permute(1, 0, 2, 3)

    inputs = {
        "pixel_values": perumuted_sample_test_video.unsqueeze(0),
        "labels": torch.tensor(
            [sample_test_video["label"]]
        ),  # this can be skipped if you don't have labels available.
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)

    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    return logits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_or_test", required=True, help="train or test")
    opts = parser.parse_args()

    model_ckpt = "MCG-NJU/videomae-base"  # pre-trained model from which to fine-tune
    batch_size = 8

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

    mean = image_processor.image_mean
    std = image_processor.image_std
    if "shortest_edge" in image_processor.size:
        height = width = image_processor.size["shortest_edge"]
    else:
        height = image_processor.size["height"]
        width = image_processor.size["width"]
    resize_to = (height, width)

    num_frames_to_sample = model.config.num_frames
    sample_rate = 4
    fps = 30
    clip_duration = num_frames_to_sample * sample_rate / fps
    logger.info(f"Each video is sampled at {fps} fps.")

    model_name = model_ckpt.split("/")[-1]
    new_model_name = f"{model_name}-finetuned-ucf101-subset"
    num_epochs = 4

    output_dir = f"../models/{new_model_name}"

    # Training dataset transformations.
    train_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        RandomShortSideScale(min_size=256, max_size=320),
                        RandomCrop(resize_to),
                        RandomHorizontalFlip(p=0.5),
                    ]
                ),
            ),
        ]
    )
    # Training dataset.
    train_dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(dataset_root_path, "train"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
        decode_audio=False,
        transform=train_transform,
    )

    val_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        Resize(resize_to),
                    ]
                ),
            ),
        ]
    )

    # Validation and evaluation datasets.
    val_dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(dataset_root_path, "val"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )

    test_dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(dataset_root_path, "test"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )

    if opts.train_or_test == "train":
        args = TrainingArguments(
            output_dir=output_dir,
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=False,
            metric_for_best_model="accuracy",
            push_to_hub=False,
            max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
        )

        logger.info("Initialized training arguments.")
        metric = evaluate.load("accuracy")

        trainer = Trainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=image_processor,
            compute_metrics=compute_metrics,
            data_collator=collate_fn,
        )

        logger.info("Initialized trainer.")

        train_the_model()

    else:
        trained_model = VideoMAEForVideoClassification.from_pretrained(output_dir)
        sample_test_video = next(iter(test_dataset))
        investigate_video(sample_test_video)
        logits = run_inference(trained_model, sample_test_video["video"])
        predicted_class_idx = logits.argmax(-1).item()
        print("Predicted class:", model.config.id2label[predicted_class_idx])
        print("True class:", model.config.id2label[sample_test_video["label"]])
