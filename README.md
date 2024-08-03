
# Brain Tumor Detection Using VGG16

This repository contains code for detecting brain tumors from MRI images using a convolutional neural network based on the VGG16 architecture.

## Dataset

The dataset used in this project is the [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection) from Kaggle. It contains MRI images categorized into two classes: `yes` for images with tumors and `no` for images without tumors.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/brain-tumor-detection.git
   cd brain-tumor-detection
   ```

2. Download the dataset:
   ```bash
   mkdir -p ~/.kaggle/
   mv kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection
   ```

3. Extract the dataset:
   ```python
   from zipfile import ZipFile

   file_name = "brain-mri-images-for-brain-tumor-detection.zip"
   with ZipFile(file_name,'r') as zip:
     zip.extractall()
     print('Done')
   ```

4. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare the data.
2. Train the model.
3. Plot the results.

## Results

The model is trained for 5 epochs. The training and validation accuracy, as well as loss, are plotted to visualize the performance of the model.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- [Kaggle](https://www.kaggle.com/) for providing the dataset.
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for their excellent deep learning libraries.

