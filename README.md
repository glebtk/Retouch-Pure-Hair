# Hair Denoising with DnCNN

![Cover image](https://i.imgur.com/HY6J9x3.gif)

## Introduction

In this project, we implement a DnCNN model 
to remove various types of noise from close-up images of hair.
The DnCNN model was chosen due to its effectiveness in denoising
tasks and its relatively low computational requirements compared 
to GAN-based architectures. DnCNN can efficiently eliminate noise
while preserving essential details in images, which is crucial
for hair structure visualization.

## Implementation

The project is implemented using the following technologies:

- Python 3
- PyTorch
- OpenCV
- TensorBoard
- Albumentations

The project structure is as follows:

- `/checkpoints`: Directory for saving model checkpoints during training
- `/dataset`: Directory containing the dataset
- `config.yaml`: Configuration file (hyperparameters, paths etc)
- `dataset.py`: Implementation of the custom dataset class
- `download_files.py`: Script for downloading the dataset and pretrained checkpoint in one click
- `inference.py`: Script for running inference on a sample image
- `models.py`: Implementation of the DnCNN model class
- `optimization.py`: Script for hyperparameter search (model, training)
- `requirements.txt`: Requirements of the project
- `train.py`: Main training script
- `transforms.py`: Implementation of image transformation class with Albumentations
- `utils.py`: Auxiliary functions for checkpoint loading, image post-processing, and setting seeds

### Requirements:

- tqdm
- numpy
- torch
- PyYAML
- pandas
- Pillow
- matplotlib
- torchvision
- scikit-learn
- scikit-image
- torchsummary
- opencv-python
- albumentations

## Dataset

The dataset used in this project is a custom dataset consisting 
of close-up, three-channel hair images with various types of
noise. The images are 256x256 pixels in size.

![Dataset examples](https://i.imgur.com/FRzvQ8J.png)


## Training process

The model was trained using the following 
parameters, which were determined through 
hyperparameter optimization:

- `num_layers: 7`
- `num_features: 128`
- `batch_size: 32`
- `learning_rate: 0.0003`
- `weight_init: none`

The training process was stopped at the 78th epoch. The highest PSNR (Peak Signal-to-Noise Ratio) was achieved at the 72nd epoch, with a value of **29.32**. The highest SSIM (Structural Similarity Index Measure) was achieved at the 75th epoch, with a value of **0.819**.

<details>
    <summary><strong>What are these metrics?</strong></summary>

- **PSNR (Peak Signal-to-Noise Ratio):** PSNR is a measure of the quality of image reconstruction, comparing the original image with the reconstructed one. A higher PSNR value indicates better reconstruction quality, meaning the reconstructed image is closer to the original one.


- **SSIM (Structural Similarity Index Measure):** SSIM is a metric used to evaluate the structural similarity between two images. It measures the similarity between the original and reconstructed images by taking into account luminance, contrast, and structure. A higher SSIM value indicates a better match in terms of structure, meaning the reconstructed image is more similar to the original one.


- **MSE (Mean Squared Error):** MSE is the average of the squared differences between the original and reconstructed images. It measures the difference in intensity values between the two images. A lower MSE value indicates that the reconstructed image is closer to the original image in terms of pixel intensities.
</details>

On average, the time spent on one epoch of training
was **78 seconds** (Nvidia Tesla T4).

The graphs below display the training process, 
showcasing the improvement in **PSNR** and **SSIM** 
over time. The model was tested on a test set, which 
comprised 20% of the original dataset. Additionally,
the graphs also show the Mean Squared Error (MSE) metric values.

![Graphs](https://i.imgur.com/MgL5iYC.png)



## Results

The DnCNN model successfully removes noise from 
the hair images while preserving essential hair 
structure details. The resulting images demonstrate 
improved quality compared to the noisy input images. 


![Results](https://i.imgur.com/NuZLJZc.png)

Nevertheless, there are still issues with catastrophically
damaged photos. On such images, the model may hallucinate

## Colab Notebook

A Colab notebook is available to quickly test the pre-trained model inference:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1la4U90GCeuPvlpNXx6mnRjqQN7mQItEP?usp=sharing)

## Contact Information

If you have any questions or feedback, please feel free to contact me:

[![Mail](https://i.imgur.com/HILZFT2.png)](mailto:tutikgv@gmail.com)
**E-mail:**
[tutikgv@gmail.com](mailto:tutikgv@gmail.com) <br>

[![Telegram](https://i.imgur.com/IMICyTA.png)](https://t.me/glebtutik)
**Telegram:**
https://t.me/glebtutik <br>

## References

- Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L. (2017). Beyond a Gaussian denoiser: Residual learning of deep CNN for image denoising. IEEE Transactions on Image Processing, 26(7), 3142-3155.

- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.
