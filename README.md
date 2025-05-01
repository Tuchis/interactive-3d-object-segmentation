# Promptable 3D Object Segmentation

This is a repository for the bachelor thesis "Interactive Object Segmentation in 3D Using 2D Foundation Models" by [Vladyslav Humennyy](https://github.com/Tuchis) under the supervision of [Ruslan Partsey](https://github.com/rpartsey).

> This thesis explores capabilities of 2D foundational models for the task of interactive 3D object segmentation. Traditional 3D object segmentation often require extensive training dataset, and may struggle to generalize to other environments. Our work explores capabilities of usage of foundation 2D segmentation model to project segmentation output to 3D point cloud, enablign human-in-the-loop interaction for improved accuracy. We implement a SAM-based 3D segmentation pipeline and develop an interactive web tool for segmentation. Our experiments demonstrate, that 2D foundation models can achieve competitive performance compared to other learning-based models, particularly with help of a human anotator. We discuss challenges such as evaluation prompt selection strategies and false positive segmentations and suggest future directions for that research. Our findings highlight the potential of 2D foundation models for flexible interactive 3D object segmentation without the need of additional training.

Here is a demo of it:

![Demo](docs/demo.gif)

## Overview of the repository:

## Structure

The repository is organized into several folders, each containing a different part of the thesis. The current structure is as follows:

- `./AGILE3D` - Contains [AGILE3D](https://github.com/ywyue/AGILE3D) related code, which is used to evaluate the performance of the proposed method.
- `./interactive-tool` - Contains the code for the interactive tool backend, which is used as a layer between the frontend and our segmentation model.
- `./interactive-tool-frontend` - Contains the code for the interactive tool frontend, which is used to visualize the segmentation results.
- `./SAMPro3D` - Contains the code of our segmentation model, which builds on top of the [SAMPro3D](https://github.com/GAP-LAB-CUHK-SZ/SAMPro3D) implementation.
- `./docs` - Contains the documentation for the project.
- `./scannet_pp` - Contains the code for the preprocessing of the [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) dataset.

## Visualization

The visualization tool consists of two components:

1. Backend server (in `interactive-tool` folder)
2. Frontend application (in `interactive-tool-frontend` folder)

To run the visualization:

1. Start the backend server on your remote machine following instructions in `interactive-tool/README.md`
2. Set up port forwarding from the remote server to your local machine
3. Launch the frontend application locally following instructions in `interactive-tool-frontend/README.md`

The frontend will automatically connect to the backend through the forwarded port, allowing you to interact with the visualization tool.

## Setup & Installation

### AGILE3D

To run AGILE3D, you need to install the environment as shown in their instructions.

**Note for installation:** AGILE3D is best run on CUDA version 11.6. If you have multiple CUDA versions on your system, you can specify the correct version by adding the following to your `.bashrc` file:

```bash
export PATH=/usr/local/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH
```

Additionally, you need to specify the correct `CUDA_HOME` during installation.

After successful installation, you have to download the dataset ([link 1](https://drive.google.com/file/d/1cqWgVlwYHRPeWJB-YJdz-mS5njbH4SnG/view), [link 2](https://polybox.ethz.ch/index.php/s/vW5GtSDlf86k2Td)) and the [checkpoint](https://polybox.ethz.ch/index.php/s/RnB1o8X7g1jL0lM):

```bash
# Download datasets
curl -o data.zip https://polybox.ethz.ch/index.php/s/vW5GtSDlf86k2Td/download
unzip data.zip
rm data.zip
mv agile3d_data/ data/
cd data
unzip KITTI360.zip -d KITTI360
unzip S3DIS.zip -d S3DIS
unzip ScanNet.zip -d ScanNet
rm KITTI360.zip S3DIS.zip ScanNet.zip
cd ..

# Download weights
mkdir weights
cd weights
curl -o checkpoint1099.pth https://polybox.ethz.ch/index.php/s/RnB1o8X7g1jL0lM/download
cd ..
```

After this, you should be able to run everything.

### Custom SAMPro3D

To install everything correctly, you need to install the environment as shown in the instructions.

```bash
conda create -n sam-universal python=3.10 -y
conda activate sam-universal
pip install torch==1.13.0+cu118 torchvision -f https://download.pytorch.org/whl/torch_stable.html
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
cd SAMPro3D/libs/pointops
TORCH_CUDA_ARCH_LIST="7.5 8.0" python setup.py install
pip install git+https://github.com/facebookresearch/segment-anything.git
cd ../../../..
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd ../promptable-3d-object-segmentation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
export CUDA_HOME=/usr/local/cuda-11.8
pip install pip==22.3
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
pip install -r SAMPro3D/requirements.txt
pip install -r interactive-tool/requirements.txt
pip install wandb
pip install "numpy<2"
```

After this, you should be able to run interactive tool and Evaluation

## How to Run

### Interactive Tool

To run the interactive tool, you need to start the backend server and the frontend application. To run the frontend application, you have to run the following command:

```bash
cd interactive-tool-frontend
npm install
npm run dev
```

To run the backend server, you have to run the following command:

```bash
cd interactive-tool
python app.py
```

! To run backend, you need to have setup ScanNet++ dataset. That is private repository, so I can't provide data for it, but you can contact me to setup backend for you on the server. With that, you can run frontend on your local machine, with all functionality. Contact me by email or any other way you can find.

### Evaluation

To run the evaluation, you can use scripts from `AGILE3D` repository. You can run it with following command:

```bash
cd AGILE3D
bash scripts/eval_single_scannetpp_sam.sh
```

Results of evaluation are saved in `AGILE3D/results/ScanNetPP_short_single` folder.

## Results

| **#** | **Experiment**                     | **Average IoU (%)** |
|-------|------------------------------------|----------------------|
| 1.    | SAM                                | 31.1                |
| 2.    | SAM w/ negative background         | 48.4                |
| 3.    | SAM 2                              | 29.4                |
| 4.    | SAM 2 w/ negative background       | 47.8                |
| 5.    | AGILE3D                            | **63.2**            |

**Table:** Quantitative comparison of segmentation methods average IoU.

| **#**  | **Reached cases (%)** 50 | 65  | 80  | 85  | 90  | **Clicks needed** 50 | 65  | 80  | 85  | 90  |
|--------|--------------------------|-----|-----|-----|-----|------------------------|-----|-----|-----|-----|
| 1. SAM                     | 27.9 | 16.2 | 4.5  | 3.4  | 1.7  | 1.2  | 1.5  | 1.3  | 1.5  | 1.3  |
| 2. SAM w/ neg. bg.         | 50.2 | 34.2 | 13.4 | 7.6  | 3.3  | 1.6  | 2.0  | 2.5  | 2.5  | 2.5  |
| 3. SAM 2                   | 25.0 | 13.9 | 3.3  | 3.3  | 1.7  | 1.2  | 1.4  | 1.3  | 1.3  | 2.0  |
| 4. SAM 2 w/ neg. bg.       | 46.7 | 33.3 | 16.7 | 7.8  | 3.9  | 1.5  | 1.7  | 1.9  | 1.9  | 2.9  |
| 5. AGILE3D                 | 72.6 | 52.0 | 31.3 | 25.1 | 14.5 | 2.0  | 2.0  | 2.1  | 2.4  | 2.8  |

**Table:** Quantitative comparison of segmentation methods based on the number of test cases reaching a target IoU and the number of clicks required.

| **#**  | 1 click | 2 clicks | 3 clicks | 5 clicks |
|--------|---------|----------|----------|----------|
| 1. SAM                  | 27.8    | 20.0     | 19.2     | 15.1     |
| 2. SAM w/ neg. bg.      | 36.0    | 38.8     | 40.4     | 39.2     |
| 3. SAM2                 | 26.3    | 20.3     | 19.1     | 15.3     |
| 4. SAM2 w/ neg. bg.     | 37.2    | 38.7     | 40.4     | 40.3     |
| 5. AGILE3D              | 39.9    | 47.5     | 53.0     | 60.3     |

**Table:** Quantitative comparison of segmentation methods on average IoU after *n* user clicks.

## Usage Notes

- If you need to run anything or access to the dataset - contact me.
