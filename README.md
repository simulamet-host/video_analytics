# E2E Video and Image Preprocessing for DL: Domain Independent Pipeline

[![Python 3.8](https://img.shields.io/badge/python-=%3E3.8-blue.svg)](https://www.python.org/downloads/release/python-3816/)
[![Test](https://github.com/simulamet-host/video_analytics/actions/workflows/pytest.yml/badge.svg)](https://github.com/simulamet-host/video_analytics/actions/workflows/pytest.yml)
[![codecov.io](https://codecov.io/github/simulamet-host/video_analytics/coverage.svg?branch=master)](https://codecov.io/github/simulamet-host/video_analytics?branch=master)


[![Pylint](https://github.com/simulamet-host/video_analytics/actions/workflows/pylint.yml/badge.svg)](https://github.com/simulamet-host/video_analytics/actions/workflows/pylint.yml)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/simulamet-host/video_analytics/issues)
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://simulamet-host.github.io/video_analytics/e2evideo.html)
 [![Documentation](https://img.shields.io/badge/Documentation-Documentation-green)](https://faiga91.github.io/e2evideo/feature_extractor.html)


## 📖 Description
e2evideo is a versatile Python package designed for video and image pre-processing and analysis 🎥📸. It comprises domain-independent modules that can be customized to suit specific tasks in various fields of computer vision.	

![package overview](./figures/overview.png)

## 🛠️ Installation
To install e2evideo, clone the Git repository, navigate to the directory, and run:
```bash
pip install .
```

# 🚀 Features :
- 🎞️ **Video Pre-processing:** Supports various video formats, frame extraction, and background subtraction.

![video_preprocessing](figures/video_preprocessor.png)

- 🖼️ **Image Pre-processing:** Converts images to greyscale, resizes images, and structures videos into compressed arrays.

![image_preprocessing](figures/image_preprocessor.png)

- 🧠 **Feature Extraction:** Includes a pre-trained ResNet18 model and DINOv2 for image embedding extraction.

![feature_extractor](figures/feature_extractor.png)

## 💻 Usage
Import the package and utilize its modules as required:
```python
import e2evideo
# Your code here
```

## 📚 Documentation
For more detailed instructions and examples, refer to the [Documentation](https://faiga91.github.io/e2evideo/feature_extractor.html).

## 🤝 Contributing
Contributions to E2Evideo are welcome! If you would like to contribute, please fork the repository and create a pull request.

![contibute](figures/contribute.png)

## 📜 License
E2Evideo is available under the MIT License 📄.

## 📃 Citation
For academic use, please cite the package as follows:
```bibtex
@inproceedings{10.1007/978-3-031-53302-0_19,
	author = {Alawad, Faiga and Halvorsen, P{\aa}l and Riegler, Michael A.},
	booktitle = {MultiMedia Modeling},
	editor = {Rudinac, Stevan and Hanjalic, Alan and Liem, Cynthia and Worring, Marcel and J{\'o}nsson, Bj{\"o}rn Þ{\'o}r and Liu, Bei and Yamakata, Yoko},
	isbn = {978-3-031-53302-0},
	pages = {258--264},
	publisher = {Springer Nature Switzerland},
	title = {E2Evideo: End to End Video and Image Pre-processing and Analysis Tool},
	year = {2024}}
```


