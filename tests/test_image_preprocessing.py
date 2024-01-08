import numpy as np
import pytest
import cv2


@pytest.fixture
def image_folder(tmpdir):
    # create temporary image folder
    tmpdir.mkdir("images")
    # define colors for test images
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    # Loop over the colors and create an image for each color
    for i, color in enumerate(colors):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:, :, :] = color
        cv2.imwrite(str(tmpdir.join(f"images/image_{i}.jpg")), img)

    # add clearnup function
    def cleanup():
        # remove the temporary folder
        tmpdir.remove()

    # return the temporary folder
    yield tmpdir.join("images")
    # call the cleanup function
    cleanup()


class TestImagesPreprocessor:
    """unit tess for image_preprocessing.py"""

    pass
