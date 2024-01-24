import numpy as np
import pytest
import cv2


@pytest.fixture
def image_folder(tmpdir):
    tmpdir.mkdir("images")
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i, color in enumerate(colors):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:, :, :] = color
        cv2.imwrite(str(tmpdir.join(f"images/image_{i}.jpg")), img)

    def cleanup():
        tmpdir.remove()

    yield tmpdir.join("images")
    cleanup()
