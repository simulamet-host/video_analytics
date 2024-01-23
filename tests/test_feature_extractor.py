"""Unit tests for FeatureExtractor class"""
import os
import pytest
from unittest.mock import patch
import argparse
import e2evideo.feature_extractor as e2e_fe


@pytest.fixture
def output_folder(image_folder):
    """Fixture to create output folder"""
    output_path = os.path.join(os.path.dirname(image_folder), "output")
    os.makedirs(output_path, exist_ok=True)
    yield output_path


@pytest.fixture
def mock_args(image_folder, output_folder):
    with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
        mock_parse_args.return_value = argparse.Namespace(
            input_path=str(image_folder),
            output_path=output_folder,
            feature_extractor="img2vec",
        )
        yield


def test_main_function(mock_args, output_folder):
    """Test main function"""
    e2e_fe.main()
    assert "vec_df.csv" in os.listdir(str(output_folder))


def test_feature_extractor(image_folder, output_folder):
    """Test FeatureExtractor class"""
    # create temp output folder
    feature_extractor_config = e2e_fe.FeatureExtractorConfig(
        input_path=image_folder, output_path=output_folder
    )
    feature_extractor = e2e_fe.FeatureExtractor(feature_extractor_config)
    file_names, feature_vec = feature_extractor.extract_dinov2_features()

    file_names = [os.path.join(output_folder, file_name) for file_name in file_names]

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
