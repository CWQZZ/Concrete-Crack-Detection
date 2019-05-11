# Concrete-Crack-Detection

### UPDATE(5/11/2019): This is the updated PyTorch Version. The deprecated TensorFlow version is in the tensorFlow branch.
This repository contains the code for crack detection in concrete surfaces. It is a PyTorch implementation of the paper by by Young-Jin Cha and Wooram Choi - "Deep Learning-Based Crack Damage Detection Using Convolutional Neural Networks".

![CNN_Archi](https://user-images.githubusercontent.com/32497274/34506710-30363d94-effd-11e7-864a-bec0d7153721.PNG)

<!-- The model acheived 85% accuracy on the validation set. A few results are shown below -
![results](https://user-images.githubusercontent.com/32497274/34510394-8e4ec3e6-f021-11e7-8a70-394219f76ff2.PNG)
 -->

MATLAB was used to prepare the data. Regions of Interest were sliced into smaller 128 x 128 pixel images and used for training - 

![roi](https://user-images.githubusercontent.com/32497274/34510417-c3207466-f021-11e7-9bf7-c91c034a70be.PNG)

Dependencies required-<br />
- PyTorch<br />
- MatPlotlib<br />
- Numpy <br />
- <b>Dataset</b> -The data set can be downloaded from [this link]( https://drive.google.com/file/d/1kC60RGO3rcScVk7HY-s7tTMJeMbADfh1/view?usp=sharing)<br />
   
To train the network run the command with the following arguments:<br />
`python main.py --mode train`<br />

The models are saved in `--save_dir` with a time and date stamp. For more information about the arguments, please refer to `main.py`. <br />
Please note that the dataset for training this is fairly small and limited. If you want to increase the performance, I would recommend obtaining more data from the MATLAB script. <br />

To test the trained network on any number of images, run the model with the following arguments:
`python main.py --mode test --test_dir PATH_TO_FOLDER_CONTAINING_IMAGES`


## TODO:

 - [ ] Upload saved model file
