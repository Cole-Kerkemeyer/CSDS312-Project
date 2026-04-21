# Tumor Classification and Segmentation Model

This model uses a [PyTorch](https://github.com/pytorch/pytorch) convolutional neural network (CNN) along with UNET architecture to conduct classification and segmentation of tumor images. The project uses advanced deep learning techniques to accurately identify and delineate tumor regions in medical imaging data. All data is from [BRISC 2025](https://www.kaggle.com/datasets/briscdataset/brisc2025), which is publicly available from Kaggle.

## Project Content
 
```
CSDS312-PROJECT/
├── Classification/
│   ├── Data/
│   ├── brain_tumor_model.pth           # Pre-trained classification model
│   ├── cnn.py                          # Local CNN training
│   ├── job.slurm                       # Training script for HPC
│   └── modelTest.py                    # Model evaluation utilities
├── Segmentation/
│   ├── .gitignore
│   ├── model.py                        # UNET segmentation model
│   ├── train_unet.slurm                # Training script for HPC
│   └── train.py                        # Local training script
├── .gitignore
├── config.json                         # Project configuration
├── llm_client.py                       # OpenAI-compatible LLM client
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

### LLM Integration

This project now includes a lightweight OpenAI-compatible client in `llm_client.py`.

1. Set environment variables:

    ```bash
    # Required
    set OPENAI_API_KEY=your_api_key

    # Optional: default model
    set OPENAI_MODEL=Qwen2.5-7B-Instruct

    # Optional: OpenAI-compatible provider endpoint (example: Alibaba DashScope)
    set OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    ```

2. Quick test:

    ```bash
    python llm_client.py
    ```

3. Use in your scripts

    ```python
    from llm_client import LLMClient

    llm = LLMClient(model="Qwen2.5-7B-Instruct")
    result = llm.chat("Explain overfitting in 3 bullet points.")
    print(result)
    ```

## Results

### Classification Training and Accuracy

CNN Training:

CNN Accuracy:
```bash
Classifying Images: 100%|███████████████████████████| 1000/1000 [00:15<00:00, 64.24it/s]
Overall Accuracy: 99.00%
```

### Segmentation Training and Accuracy 

