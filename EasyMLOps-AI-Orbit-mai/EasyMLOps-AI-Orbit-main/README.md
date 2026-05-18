# easyMLOps (Gradio-based MLOps Framework)

This repository provides the implementation of a Gradio-based MLOps framework
used in the accompanying paper. The system supports interactive dataset upload,
training monitoring, performance monitoring, and model comparison via a
web-based Gradio interface.

## Requirements
- Python 3.10+ recommended
- See requirements.txt for dependencies

## Installation
pip install -r requirements.txt

## Run (Local)
python app.py

Open your browser at:
http://127.0.0.1:7860

## Directory Structure
core/
- Core backend logic of the easyMLOps framework, including configuration handling,
  training orchestration, evaluation utilities, and file management.

ui/
- Gradio-based user interface components.
- Each submodule corresponds to a functional tab (dataset setup, training monitor,
  performance monitoring, labeling, and model comparison).

json/
- Custom JavaScript assets used to extend and control interactive behaviors
  of the Gradio interface.

tools/
- Utility scripts and helper modules for data processing, debugging,
  and auxiliary workflows.

workspace/datasets_for_labeling/
- User-uploaded datasets during runtime (not tracked in git)

workspace/base_model/
- Directory for YOLO .pt weights provided by the user (not tracked in git)

workspace/configs/
- Configuration examples (e.g., data.example.yaml)

examples/sample_project/
- Minimal example dataset for illustrating the expected data format and
  directory structure

## Weights and Datasets
Pretrained YOLO weights (.pt) and full datasets are not included in this
repository due to size and license considerations.

Please download the appropriate YOLO weights from Ultralytics and place them
under:
workspace/base_model/

Ultralytics YOLO documentation:
https://docs.ultralytics.com/models/yolov8/

## Sample Project Notice
The data provided in examples/sample_project/ is for demonstration purposes
only. It is not intended to reproduce the experimental results reported in the
paper.

Sample images are taken from the COCO dataset for demonstration purposes only.

## Live Demo (Optional)

A live demonstration of the system is available at the time of submission:
http://qisens.iptime.org:7863/

Note that the availability of the demo depends on the server status.

## License
This project is licensed under the MIT License.

