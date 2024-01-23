"""Unit tests for FeatureExtractor class"""
from e2evideo.feature_extractor import FeatureExtractor
from e2evideo.feature_extractor import FeatureExtractorConfig
import os


def test_feature_extractor(image_folder):
    """Test FeatureExtractor class"""
    # create temp output folder
    output_path = os.path.join(os.path.dirname(image_folder), "output")
    os.makedirs(output_path, exist_ok=True)

    feature_extractor_config = FeatureExtractorConfig(
        input_path=image_folder, output_path=output_path
    )
    feature_extractor = FeatureExtractor(feature_extractor_config)
    file_names, feature_vec = feature_extractor.extract_dinov2_features()

    file_names = [os.path.join(output_path, file_name) for file_name in file_names]

    expected_file_names = [
        "image_0.jpg",
        "image_1.jpg",
        "image_2.jpg",
    ]

    assert all(
        file_name.endswith(expected_name)
        for file_name, expected_name in zip(file_names, expected_file_names)
    )
    assert feature_vec.shape[0] == 3
