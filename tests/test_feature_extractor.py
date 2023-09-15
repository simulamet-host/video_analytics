"""Unit tests for FeatureExtractor class"""
import os
import shutil
import numpy as np
from PIL import Image
from e2evideo.feature_extractor import FeatureExtractor


def test_feature_extractor():
    """Test FeatureExtractor class"""

    # create input and output paths for testing
    input_path = "./input/"
    output_path = "./output/"
    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    # create colored images in input path
    img1 = Image.new("RGB", (100, 100), color="red")
    img1.save(os.path.join(input_path, "img1.png"))
    img2 = Image.new("RGB", (100, 100), color="green")
    img2.save(os.path.join(input_path, "img2.png"))
    img3 = Image.new("RGB", (100, 100), color="blue")
    img3.save(os.path.join(input_path, "img3.png"))

    # create FeatureExtractor object
    try:
        fe_extractor = FeatureExtractor(input_path, output_path)
        filenames, feature_vec = fe_extractor.extract_dinov2_features()

        # check if the number of images is correct
        print(len(filenames))
        assert len(filenames) == 3
        assert len(feature_vec) == 3
        assert isinstance(feature_vec, np.ndarray)
    except Exception as e:
        assert False
        print(e)

    # clean up
    shutil.rmtree(input_path)
    shutil.rmtree(output_path)
