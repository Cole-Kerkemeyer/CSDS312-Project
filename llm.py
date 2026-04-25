import os
import torch
import torchvision.transforms as T
from torchvision.models import resnet50
import numpy as np
from PIL import Image
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import your custom segmentation model
from segmentation.model_brisc import AttentionUNet

# ==========================================
# 1. BRISC Vision Model Inference 
# ==========================================

# BRISC Class Mapping: 
# 0: No Tumor, 1: Glioma, 2: Meningioma, 3: Pituitary Tumor
CLASS_NAMES = {0: "No Tumor", 1: "Glioma", 2: "Meningioma", 3: "Pituitary Tumor"}

def run_brisc_classifier(t1_image_path, checkpoint_path="checkpoints/classifier_best.pth"):
    """
    Loads a ResNet50 classification model, preprocesses the T1 image, 
    and predicts the tumor class.
    """
    print(f"Running Classification on {t1_image_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Initialize Model Architecture (Assuming a ResNet50 for 4 classes)
    model = resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 4) 
    
    # 2. Load Weights safely (strips 'module.' if saved via DataParallel)
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict)
    else:
        print(f"WARNING: Classifier checkpoint not found at {checkpoint_path}. Using random weights!")
        
    model.to(device)
    model.eval()

    # 3. Preprocess Image (Standard ImageNet transforms used for ResNet)
    image = Image.open(t1_image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device) # Shape: (1, 3, 256, 256)

    # 4. Run Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        
    return CLASS_NAMES[predicted_idx.item()]


def run_brisc_segmenter(t1_image_path, checkpoint_path="checkpoints/attention_unet_best.pth"):
    """
    Loads the Attention U-Net, predicts the tumor mask, 
    and calculates the physical area of the tumor.
    """
    print(f"Running Segmentation on {t1_image_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Initialize Model Architecture
    model = AttentionUNet(in_channels=3, num_classes=1)
    
    # 2. Load Weights safely
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict)
    else:
        print(f"WARNING: Segmenter checkpoint not found at {checkpoint_path}. Using random weights!")
        
    model.to(device)
    model.eval()

    # 3. Preprocess Image (Must match Albumentations validation logic EXACTLY)
    # Using torchvision here to avoid needing to import albumentations in inference,
    # mathematically equivalent to Albumentations ToTensorV2 and Normalize.
    image = Image.open(t1_image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    # 4. Run Inference
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float() # Binary mask (1 for tumor, 0 for background)
        
    # 5. Calculate Physical Area
    tumor_pixels = preds.sum().item()
    
    # Since BRISC converts original MRIs to 256x256 JPEGs, physical DICOM spacing is lost.
    # For clinical pipelines, you define a calibration factor (e.g., 1 pixel = 0.8 mm²)
    PIXEL_TO_MM2_RATIO = 1.0 
    
    tumor_area_mm2 = tumor_pixels * PIXEL_TO_MM2_RATIO
    return round(tumor_area_mm2, 2)

def free_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

# ==========================================
# 2. LLM Report Generation
# ==========================================
def generate_radiology_report(patient_info, tumor_class, tumor_area):
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    print(f"\nLoading LLM ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype="auto", 
        device_map="auto" 
    )

    system_prompt = """You are an expert, board-certified neuroradiologist. 
Your task is to write a formal clinical MRI brain scan report based on AI-extracted findings from a T1-weighted MRI.

Your report MUST adhere to this structure:
1. CLINICAL INDICATION: Patient history.
2. TECHNIQUE: Note that a single-plane T1-weighted MRI was reviewed.
3. FINDINGS: Detail the tumor classification and cross-sectional area. 
4. IMPRESSION: A concise conclusion summarizing the diagnosis.

Maintain a highly professional, objective clinical tone. Do not invent findings."""

    if tumor_class == "No Tumor" or tumor_area == 0:
        findings_text = "No focal lesions, masses, or abnormalities detected. Volume of interest mask measures 0 mm²."
    else:
        findings_text = f"A mass consistent with a {tumor_class} is identified. The cross-sectional area of the segmented lesion is approximately {tumor_area} mm²."

    user_prompt = f"""
Please generate the radiology report using the following data:

[PATIENT INFO]
Name: {patient_info['name']}
Age: {patient_info['age']}
Gender: {patient_info['gender']}
Symptoms: {patient_info['symptoms']}

[AI FINDINGS]
{findings_text}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(llm.device)

    print("\n" + "="*50)
    print("GENERATING SYNTHETIC RADIOLOGY REPORT")
    print("="*50)
    
    generated_ids = llm.generate(
        **model_inputs,
        max_new_tokens=400,
        temperature=0.2, 
        do_sample=True
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    print("="*50)

# ==========================================
# 3. Execute Pipeline
# ==========================================
if __name__ == "__main__":
    # Simulated database fetch for the patient
    patient_data = {
        "name": "Sarah Jenkins",
        "age": 42,
        "gender": "Female",
        "symptoms": "Progressive vision loss in the right eye and mild chronic headaches."
    }
    
    # Path to the actual T1 image file you want to evaluate
    target_scan = "./data/brisc2025/segmentation_task/test/images/test_001.jpg"
    
    # Make sure you create a "checkpoints" folder and place your trained .pth files there!
    
    # 1. Run Computer Vision Models
    predicted_class = run_brisc_classifier(target_scan, checkpoint_path="checkpoints/classifier_best.pth")
    predicted_area = run_brisc_segmenter(target_scan, checkpoint_path="checkpoints/attention_unet_best.pth")
    
    # 2. Flush GPU to make room for the Large Language Model
    free_gpu_memory()
    
    # 3. Generate Clinical Report
    generate_radiology_report(patient_data, predicted_class, predicted_area)