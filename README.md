# Decoupling Layout from Glyph in Online Chinese Handwriting Generation
## 🎉🎉🎉 This work has been accepted by ICLR2025 🎉🎉🎉
## Data and Enviroment
The data can be download at "https://huggingface.co/datasets/Immortalman12/OLHWD/tree/main".

The description of each data file can be found there.

The required environment is particularly simple, only basic libraries such as torch, numpy, matplotlib, and math are needed.

## How to build a framework for Online Chinese Handwriting Generation
**It consists of two steps:**  
1. Single Character Generation
2. Layout Generation

### Single Character Generation
| File name       | Main functions                                                         |
|-----------------|------------------------------------------------------------------------|
|`basemodel.py`   | Network Architecture (1D-Unet)                                         |
|`unet_ddpm.py`   | Diffusion Model Implementation                                         |
|`train_ddpm.py`  | Train and test the diffusion model                                     |
|`style_classifier.py` | Style Encoder                                                     |
|`evaluate_char.py`    | Evaluate the character generator                                  |  
|`./evaluate_char/char_classifier.py` | Train the content classifier for evaluation        |
|`./evaluate_char/style_classifier.py` | Train the style classifier for evaluation         |

> 1. Download the data into `./datas`
> 2. Train the diffusion model  `train_ddpm.py`  
> 3. For evaluation, train the content and style classifiers 
> 4. Evaluate and Finished `evaluate_char.py` 


### Layout Generation
| File name       | Main functions                                                         |
|-----------------|------------------------------------------------------------------------|
|`./bonibox_gen/count.py`   | Count the border information of each character type in the dataset|
|`./bonibox_gen/char_boxes.npy`   | The border information of each character type  （Use Gaussian Distribution|
|`./bonibox_gen/train_box_generator.py`  | Train the layout planner LSTM module            |
|`./bonibox_gen/simpebox.py` | Generate layout utilizing Gaussian distribution for each character |
|`./bonibox_gen/generatebox.py` | Generate layout utilizing LSTM network|
|`./bonibox_gen/evaluate_box.py` | Evaluate the generated layout   |

> 1. Get the data `./bonibox_gen/char_boxes.npy`
> 2. Train the layout model  `./bonibox_gen/train_box_generator.py`
> 3. Evaluate and Finished `./bonibox_gen/evaluate_box.py` 

### Full Text Line Generation
> Utilizing `generate_line.py`
