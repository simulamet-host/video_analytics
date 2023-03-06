# E2E Video and Image Preprocessing for DL: Domain Independent Pipeline

[![Python 3.8](https://img.shields.io/badge/python-=%3E3.8-blue.svg)](https://www.python.org/downloads/release/python-3816/)
[![Test](https://github.com/simulamet-host/video_analytics/actions/workflows/testing.yml/badge.svg)](https://github.com/simulamet-host/video_analytics/actions/workflows/testing.yml)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/simulamet-host/video_analytics/issues)
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://simulamet-host.github.io/video_analytics/e2evideo.html) 

This project provides general modules for video and images preprocessing and feature extraction.
Those are domain independent, but can be tailored towards specific problems by passing optional attributes to the different modules. :star2:	

![system design](System%20Pipeline.drawio.svg)

# Features :rocket:
E2Evideo provides the following features:
- :white_check_mark: Video and image loading from disk or from a URL :file_folder:	
- :white_check_mark: Video and image resizing, cropping, and normalization :camera_flash:	
- :white_check_mark: Video and image frame extraction :camera:	
- (In-progress) Object detection and tracking in video and images :film_projector:
- (TODO) Face detection and recognition in video and images :female_detective:
- (In-progress) Deep feature extraction using pre-trained models (e.g. VGG16, ResNet50) :robot:	
- (TODO) Data augmentation techniques for video and image data :framed_picture:	


# Installation :computer:	
To install E2Evideo, you can use the following command after git Clone this Repo, then run the following in the main directory:

``
pip install .
`` 


# Contributing :busts_in_silhouette: 
Contributions to E2Evideo are welcome! If you would like to contribute, please fork the repository and create a pull request.

# License :page_facing_up:	
E2Evideo is licensed under the MIT License. See LICENSE for more information.

