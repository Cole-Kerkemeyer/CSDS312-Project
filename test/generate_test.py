import os
import json
import random

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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all jpgs in the directory
    all_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    # Shuffle to get a random mix of tumors and planes
    random.shuffle(all_files)
    selected_files = all_files[:num_cases]

    print(f"Generating {len(selected_files)} synthetic patient records...\n")

    for i, filename in enumerate(selected_files):
        # Example filename: brisc2025_test_00001_gl_ax_t1.jpg
        parts = filename.replace('.jpg', '').split('_')
        
        # Safely extract codes based on the BRISC naming convention
        try:
            tumor_code = parts[3] # 'gl', 'me', 'pi', 'no'
            plane_code = parts[4] # 'ax', 'co', 'sa'
        except IndexError:
            print(f"[!] Skipping {filename}: Unrecognized naming format.")
            continue

        if tumor_code not in SYMPTOM_MAP or plane_code not in PLANE_MAP:
            continue

        patient_id = f"BRISC-PT-{1000 + i}"
        plane_name = PLANE_MAP[plane_code]
        symptom = random.choice(SYMPTOM_MAP[tumor_code]['symptoms'])

        # Build the JSON payload
        patient_data = {
            "patient_id": patient_id,
            "name": f"Test Patient {i+1}",
            "age": random.randint(30, 75),
            "gender": random.choice(["Male", "Female"]),
            "scan_plane": plane_name,  # <--- We save this so the VLM knows!
            "symptoms": symptom,
            "scan_path": os.path.join(image_dir, filename),
            "generated_report": None
        }

        output_path = os.path.join(output_dir, f"{patient_id}_{tumor_code}.json")
        
        with open(output_path, 'w') as f:
            json.dump(patient_data, f, indent=4)

        print(f"[+] Created {patient_id} -> {tumor_code.upper()} ({plane_name})")

if __name__ == "__main__":
    # Point this to your actual test images folder
    IMAGE_DIRECTORY = "data/brisc2025/segmentation_task/test/images"
    OUTPUT_DIRECTORY = ".test/patient_intake/"
    
    generate_patient_jsons(IMAGE_DIRECTORY, OUTPUT_DIRECTORY, num_cases=20)