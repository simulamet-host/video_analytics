# E2E Video and Image Preprocessing for DL: Domain Independent Pipeline

[![Python 3.8](https://img.shields.io/badge/python-=%3E3.8-blue.svg)](https://www.python.org/downloads/release/python-3816/)
[![Test](https://github.com/simulamet-host/video_analytics/actions/workflows/testing.yml/badge.svg)](https://github.com/simulamet-host/video_analytics/actions/workflows/testing.yml)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/simulamet-host/video_analytics/issues)
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://simulamet-host.github.io/video_analytics/E2Evideo.html) 

This project provides general modules for video and images preprocessing and feature extraction.
Those are domain independent, but can be tailored towards specific problems by passing optional attributes to the different modules.

![system design](System%20Pipeline.drawio.svg)

# Features
E2Evideo provides the following features:
- Video and image loading from disk or from a URL
- Video and image resizing, cropping, and normalization
- Video and image frame extraction
- (In-progress) Object detection and tracking in video and images
- (TODO) Face detection and recognition in video and images
- Deep feature extraction using pre-trained models (e.g. VGG16, ResNet50)
- (TODO) Data augmentation techniques for video and image data


# Installation
To install E2Evideo, you can use the following command after git Clone this Repo, then run the following in the main directory:

``pip install .`` 

# Contributing
Contributions to E2Evideo are welcome! If you would like to contribute, please fork the repository and create a pull request.

# License
E2Evideo is licensed under the MIT License. See LICENSE for more information.

