import os
import matplotlib.pyplot as plt
import cv2


def display_video_frames(frames, n_rows):
    n_cols = len(frames) // n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5))
    axes_flat = axes.flatten()
    for ax, frame in zip(axes_flat, frames):
        ax.imshow(frame)
        ax.axis("off")
        for i in range(len(frames), len(axes_flat)):
            axes_flat[i].axis("off")
    plt.show()


def display_foreground_mask(image_path):
    orig_image = cv2.imread(image_path)
    if orig_image is None:
        print(f"Failed to load image from {image_path}")
        return

    directory, image_name = os.path.split(image_path)
    mask_image_name = f"fgMask_{image_name}"
    mask_image_path = os.path.join(directory, mask_image_name)

    # Read the foreground mask image
    mask_image = cv2.imread(mask_image_path)
    if mask_image is None:
        print(f"Failed to load mask image from {mask_image_path}")
        return

    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)

    # Display the images
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    axes[0].imshow(orig_image)
    axes[0].set_title("Original Frame")
    axes[0].axis("off")

    axes[1].imshow(mask_image)
    axes[1].set_title("Foreground Mask Frame")
    axes[1].axis("off")

    plt.show()