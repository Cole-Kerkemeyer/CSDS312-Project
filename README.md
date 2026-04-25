# Tumor Classification and Segmentation Model

This model uses a [PyTorch](https://github.com/pytorch/pytorch) convolutional neural network (CNN) along with UNET architecture to conduct classification and segmentation of tumor images. The project uses advanced deep learning techniques to accurately identify and delineate tumor regions in medical imaging data. All data is from [BRISC 2025](https://www.kaggle.com/datasets/briscdataset/brisc2025), which is publicly available from Kaggle.

## Project Content
 
```
CSDS312-PROJECT/
├── Classification/
│   ├── brain_tumor_model.pth           # Pre-trained classification model
│   ├── cnn.py                          # Local CNN training
│   ├── job.slurm                       # Training script for HPC
│   └── modelTest.py                    # Model evaluation utilities
├── Segmentation/
│   ├── model.py                        # UNET segmentation model
│   ├── train_unet.slurm                # Training script for HPC
│   └── train.py                        # Local training script
├── LLM/
│   ├── config.json                     # LLM configuration
│   └── LLM_Client.py                   # LLM diagnosis client
├── .gitignore
├── requirements.txt                    # Python dependencies
└── README.md
```

## Installation & Setup
 
**Requirements:** Python 3.8+
 
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/CSDS312-Project.git
   cd CSDS312-Project
   ```
 
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Populate Data:
 
    Both classification and segmentation models require training data. Ensure your data is organized in the respective `Data/` directories, where classification data is placed in `Classification/Data/` and segmentation data in `Segmentation/` directory


## Usage
 
### Classification Model
If you want to train your own CNN model on locally or on the HPC, follow the first step, otherwise you can use the pre-trained model, located at `Classification/brain_tumor_model.pth`.

1. Local Training:
    ```bash
    python Clasification/cnn.py
    ```

2. HPC/Cluster Training:
   ```bash
   sbatch Classification/job.slurm
   ```
   
3. Testing Model:
    ```bash
    python Classification/modelTest.py
    ```

### Segmentation Model
This project uses the UNET segmentation model for precise tumor boundary delineation.

1. Local Training:
   ```bash
   python Segmentation/train.py
   ```
 
2. HPC/Cluster Training:
   For high-performance computing environments, submit the training job:
   ```bash
   sbatch Segmentation/train_unet.slurm
   ```

### LLM Diagnosis

This project now includes a lightweight OpenAI-compatible client in `llm_client.py`.
license_name: qwen-research
license_link: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE
language:
  - en
pipeline_tag: image-text-to-text
tags:
  - multimodal
library_name: transformers

This project integrates an LLM for providing diagnostic insights based on classified tumor images.

1. Run the LLM client:
   ```bash
   python LLM/LLM_Client.py
   ```

2. There will be two prompts after running:
   - Enter image file name: (e.g., path/to/image.jpg)
   - Enter your question: (e.g., "What type of tumor is this?")

3.  Then it will print the model's answer.




## Results

### Classification Training and Accuracy

CNN Training:

CNN Accuracy:
```bash
Classifying Images: 100%|███████████████████████████| 1000/1000 [00:15<00:00, 64.24it/s]
Overall Accuracy: 99.00%
```

### Segmentation Training and Accuracy 

