import os
import json
import random
import shutil

# 1. Clinical Symptom Mapping
SYMPTOM_MAP = {
    'gl': { # Glioma
        'disease': 'Glioma', 
        'symptoms': [
            "Sudden onset seizures and progressive right-sided motor weakness.",
            "Rapid cognitive decline and severe morning headaches.",
            "Progressive aphasia (difficulty speaking) and personality changes."
        ]
    },
    'me': { # Meningioma
        'disease': 'Meningioma', 
        'symptoms': [
            "Chronic, dull headaches over the past 6 months and focal cranial nerve deficits.",
            "Incidental finding after a minor fall; patient is currently asymptomatic.",
            "Slowly progressive unilateral hearing loss and mild facial numbness."
        ]
    },
    'pi': { # Pituitary
        'disease': 'Pituitary Tumor', 
        'symptoms': [
            "Bitemporal hemianopsia (loss of peripheral vision) and unexplained weight gain.",
            "Severe hormonal imbalance, galactorrhea, and chronic fatigue.",
            "Progressive vision blurring and localized frontal headache."
        ]
    },
    'no': { # No Tumor
        'disease': 'No Tumor', 
        'symptoms': [
            "Routine screening for chronic migraines; no focal neurological deficits.",
            "Post-concussion evaluation following a mild sports injury.",
            "Episodes of dizziness and vertigo; otherwise healthy."
        ]
    }
}

PLANE_MAP = {
    'ax': 'Axial',
    'co': 'Coronal',
    'sa': 'Sagittal'
}

def generate_patient_jsons(image_dir, output_dir, num_cases=20):
    # 1. Create Base Output Directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 2. Create the new Images Subdirectory
    image_output_dir = os.path.join(output_dir, "images")
    if not os.path.exists(image_output_dir):
        os.makedirs(image_output_dir)

    all_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    random.shuffle(all_files)
    selected_files = all_files[:num_cases]

    print(f"Generating {len(selected_files)} synthetic patient records...\n")

    for i, filename in enumerate(selected_files):
        parts = filename.replace('.jpg', '').split('_')
        
        try:
            tumor_code = parts[3] 
            plane_code = parts[4] 
        except IndexError:
            continue

        if tumor_code not in SYMPTOM_MAP or plane_code not in PLANE_MAP:
            continue

        patient_id = f"BRISC-PT-{1000 + i}"
        plane_name = PLANE_MAP[plane_code]
        symptom = random.choice(SYMPTOM_MAP[tumor_code]['symptoms'])

        # 3. Copy the Image to the localized folder
        source_image_path = os.path.join(image_dir, filename)
        target_image_path = os.path.join(image_output_dir, filename)
        shutil.copy2(source_image_path, target_image_path)

        # 4. Build the JSON payload (Notice scan_path points to target_image_path now)
        patient_data = {
            "patient_id": patient_id,
            "name": f"Test Patient {i+1}",
            "age": random.randint(30, 75),
            "gender": random.choice(["Male", "Female"]),
            "scan_plane": plane_name,  
            "symptoms": symptom,
            "scan_path": target_image_path, 
            "generated_report": None
        }

        output_path = os.path.join(output_dir, f"{patient_id}_{tumor_code}.json")
        
        with open(output_path, 'w') as f:
            json.dump(patient_data, f, indent=4)

        print(f"[+] Created {patient_id} -> {tumor_code.upper()} ({plane_name})")

if __name__ == "__main__":
    IMAGE_DIRECTORY = "./data/brisc2025/segmentation_task/test/images/"
    OUTPUT_DIRECTORY = "./test/patient_intake/"
    
    generate_patient_jsons(IMAGE_DIRECTORY, OUTPUT_DIRECTORY, num_cases=20)