'''
Usage: python llm.py --patient-data "path to patient JSON file"

Patient JSON formatting example:

{
    "patient_id": "BRISC-PT-1000",
    "name": "Test Patient 1",
    "age": 30,
    "gender": "Female",
    "scan_plane": "Coronal",
    "symptoms": "Severe hormonal imbalance, galactorrhea, and chronic fatigue.",
    "scan_path": "data/brisc2025/segmentation_task/test/images/brisc2025_test_00856_pi_co_t1.jpg",
   s"generated_report": null
}
'''




import argparse
import json
import os
import sys
import gc
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet50
from PIL import Image
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Import your custom Attention U-Net
from Segmentation.model_brisc import AttentionUNet

# ==========================================
# 0. Safety & Utilities
# ==========================================
def enforce_weights_exist(filepath):
    """Terminates the pipeline immediately if model weights are missing."""
    if not os.path.exists(filepath):
        print(f"\n[CRITICAL ERROR] Required model weights not found at: {filepath}")
        print("Pipeline terminated to prevent generating hallucinated reports using random weights.")
        sys.exit(1)

# ==========================================
# 1. Expert Vision Models (BRISC)
# ==========================================

CLASS_NAMES = {0: "No Tumor", 1: "Glioma", 2: "Meningioma", 3: "Pituitary Tumor"}

def run_brisc_classifier(image_path, checkpoint_path="checkpoints/classifier.pth"):
    print(f"[*] Running Expert Classifier on {os.path.basename(image_path)}...")
    enforce_weights_exist(checkpoint_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Match your exact training architecture
    model = resnet50(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.4), 
        torch.nn.Linear(2048, 4)
    )
    
    # 2. Extract the weights from your custom dictionary
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Clean prefix and load
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.to(device).eval()

    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        # Calculate percentages for all classes
        probs = F.softmax(outputs, dim=1)[0]
        
    class_probabilities = {CLASS_NAMES[i]: round(probs[i].item() * 100, 2) for i in range(4)}
    return class_probabilities


def run_brisc_segmenter(image_path, checkpoint_path="checkpoints/attention_unet_best.pth"):
    print(f"[*] Running Expert Segmenter on {os.path.basename(image_path)}...")
    enforce_weights_exist(checkpoint_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionUNet(in_channels=3, num_classes=1)
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.to(device).eval()

    orig_image = Image.open(image_path).convert("RGBA").resize((256, 256))
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(orig_image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        preds = (torch.sigmoid(logits) > 0.5).float()
        
    tumor_area_mm2 = preds.sum().item() * 1.0 
    
    # Generate a visual overlay for the VLM
    mask_np = preds.squeeze().cpu().numpy()
    mask_img = Image.fromarray((mask_np * 255).astype(np.uint8)).convert("L")
    
    # Create a red overlay where the mask is positive
    red_layer = Image.new("RGBA", orig_image.size, (255, 0, 0, 128))
    mask_rgba = Image.new("RGBA", orig_image.size, (0, 0, 0, 0))
    mask_rgba.paste(red_layer, (0, 0), mask_img)
    
    # Blend the original image with the red AI mask
    overlay_image = Image.alpha_composite(orig_image, mask_rgba).convert("RGB")

    return round(tumor_area_mm2, 2), overlay_image


def free_gpu_memory():
    print("[*] Unloading Expert Models and freeing VRAM...")
    gc.collect()
    torch.cuda.empty_cache()

# ==========================================
# 2. Vision-Language Model (VLM) Engine
# ==========================================

def generate_vlm_report(patient_data, class_probs, tumor_area, overlay_image):
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    print(f"\n[*] Booting Multimodal Agent ({model_name})...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)

    # Format the probability distribution nicely
    prob_text = "\n".join([f"  - {k}: {v}%" for k, v in class_probs.items()])
    
    findings_text = f"ML Classification Probabilities:\n{prob_text}\n\nML Segmentation Area: {tumor_area} mm²."

    # 2. UPDATED PROMPT: Acknowledge both images and add conflict resolution rules
    system_prompt = """You are an expert, board-certified neuroradiologist. Your task is to synthesize clinical history, visual MRI assessment, and quantitative ML data into a formal, medically accurate radiology report.

CRITICAL REASONING RULES:
1. You are the final clinical judge. If ML classification probabilities contradict the ML segmentation area (e.g., high probability of 'No Tumor' but a large mask is drawn), or if the patient symptoms contradict the scan, note the discrepancy and recommend clinical correlation. 
2. Do not invent or hallucinate findings, patient symptoms, or medical advice not supported by the provided data.

REPORT STRUCTURE AND STYLE REQUIREMENTS:
1. CLINICAL INDICATION: A brief summary of the patient's history and symptoms.
2. TECHNIQUE: State that a T1-weighted MRI was evaluated alongside ML quantitative segmentation and classification tools.
3. FINDINGS: This must be a COMPREHENSIVE, highly descriptive, and detailed paragraph. 
   - Describe the exact anatomical location based on the unaltered scan and the red mask.
   - Describe the visual characteristics of the lesion (e.g., margins, homogeneity, relationship to surrounding structures, mass effect, or midline shift).
   - Explicitly integrate the ML classification probabilities (addressing differential diagnoses) and the exact measured segmentation area.
4. IMPRESSION: This must be EXTREMELY CONCISE. 
   - Provide a bottom-line clinical conclusion.
   - Use a numbered list if there are multiple distinct conclusions.
   - Do NOT discuss ML probabilities, pixel areas, or software tools here. Focus purely on the human medical diagnosis.
"""
    user_prompt = f"""Please review the two attached images:
- Image 1: The unaltered, original {patient_data.get('scan_plane', '')} T1-weighted MRI scan.
- Image 2: The same scan with a ML-generated tumor segmentation overlay.

[PATIENT HISTORY]
Name: {patient_data.get('name', 'Unknown')}
Age: {patient_data.get('age', 'Unknown')}
Symptoms: {patient_data.get('symptoms', 'None reported')}

[AI EXPERT FINDINGS]
{findings_text}

Generate the formal radiology report based on the rules provided."""

    # 3. PASS BOTH IMAGES to the VLM
    messages = [
        # 1. The System Prompt (Rules & Persona)
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        },
        # 2. The User Prompt (Images & Data)
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": patient_data["scan_path"], # The Clean Original
                },
                {
                    "type": "image",
                    "image": overlay_image,             # The AI Overlay
                },
                {
                    "type": "text", 
                    "text": user_prompt                 # The Patient Data
                },
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    print("\n" + "="*60)
    print("GENERATING MULTIMODAL RADIOLOGY REPORT")
    print("="*60)

    generated_ids = model.generate(
        **inputs, max_new_tokens=1024, temperature=0.2, do_sample=True
    )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    final_report = output_text[0]
    print(final_report)
    print("="*60)
    
    return final_report


# ==========================================
# 3. Pipeline Execution
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-data", type=str, required=True, help="Path to JSON file.")
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.patient_data):
        print(f"Error: Could not find patient JSON file at {args.patient_data}")
        sys.exit(1)
        
    with open(args.patient_data, 'r') as f:
        patient_data = json.load(f)

    scan_path = patient_data.get("scan_path")
    if not scan_path or not os.path.exists(scan_path):
        print(f"Error: Scan path '{scan_path}' is invalid or missing from JSON.")
        sys.exit(1)

    print(f"\n--- Starting Automated Pipeline for Patient: {patient_data.get('name')} ---")

    # 1. Extract Expert Data
    class_probs = run_brisc_classifier(scan_path)
    tumor_area, overlay_image = run_brisc_segmenter(scan_path)

    # 2. Flush VRAM
    free_gpu_memory()

    # 3. Multimodal Reasoning
    generate_vlm_report(patient_data, class_probs, tumor_area, overlay_image)

if __name__ == "__main__":
    main()