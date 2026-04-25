import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VLM_NAME = os.getenv("VLM_MODEL_NAME", "Qwen/Qwen2.5-VL-3B-Instruct")
DEFAULT_CLASSIFIER_CHECKPOINT = PROJECT_ROOT / "Classification" / "brain_tumor_model.pth"
DEFAULT_SEGMENTATION_NOTE = (
    "No segmentation summary was provided. The current segmentation pipeline in this "
    "project is trained on BraTS-style 2.5D multi-channel H5 volumes, so it cannot be "
    "run directly on a single JPG image without an additional inference adapter."
)


@dataclass
class ClassificationResult:
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float]

    def ranked_labels(self) -> list:
        return sorted(self.probabilities.items(), key=lambda item: item[1], reverse=True)


def resolve_model_source(model_name: str) -> str:
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return str(model_path.resolve())

    if "/" not in model_name:
        return model_name

    owner, repo = model_name.split("/", 1)
    snapshot_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{owner}--{repo}"
        / "snapshots"
    )
    if not snapshot_root.exists():
        return model_name

    snapshots = sorted(
        [path for path in snapshot_root.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return str(snapshots[0]) if snapshots else model_name


def resolve_input_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    project_candidate = (PROJECT_ROOT / raw_path).resolve()
    if project_candidate.exists():
        return project_candidate

    raise FileNotFoundError(f"Could not find file: {raw_path}")


class BrainTumorClassifier:
    def __init__(self, checkpoint_path: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.classes = checkpoint.get(
            "classes",
            ["glioma", "meningioma", "no_tumor", "pituitary"],
        )

        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(2048, len(self.classes)),
        )

        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Keep preprocessing aligned with the existing classification test script.
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.1735, 0.1735, 0.1735],
                    [0.1771, 0.1771, 0.1771],
                ),
            ]
        )

    def classify(self, image_path: Path) -> ClassificationResult:
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probabilities = torch.softmax(self.model(image_tensor), dim=1)[0].cpu().tolist()

        probability_map = {
            label: probability for label, probability in zip(self.classes, probabilities)
        }
        predicted_label, confidence = max(probability_map.items(), key=lambda item: item[1])
        return ClassificationResult(
            predicted_label=predicted_label,
            confidence=confidence,
            probabilities=probability_map,
        )


class VisionLLMClient:
    def __init__(self, model_name: str):
        self.model_name = resolve_model_source(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None

    def _load(self) -> None:
        if self.processor is not None and self.model is not None:
            return

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                local_files_only=True,
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                local_files_only=True,
            ).to(self.device)
        except Exception:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype="auto",
            ).to(self.device)

    def ask_image(self, image_path: Path, prompt: str, max_new_tokens: int = 256) -> str:
        self._load()
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        return self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]


def load_segmentation_summary(segmentation_summary_path: Optional[str]) -> str:
    if not segmentation_summary_path:
        return DEFAULT_SEGMENTATION_NOTE

    summary_path = resolve_input_path(segmentation_summary_path)
    if summary_path.suffix.lower() == ".json":
        with open(summary_path, "r", encoding="utf-8") as file:
            content = json.load(file)
        return json.dumps(content, indent=2, ensure_ascii=False)

    with open(summary_path, "r", encoding="utf-8") as file:
        content = file.read().strip()

    return content if content else DEFAULT_SEGMENTATION_NOTE


def format_probability_summary(result: ClassificationResult) -> str:
    return ", ".join(
        f"{label}: {probability:.2%}" for label, probability in result.ranked_labels()
    )


def build_project_prompt(
    question: str,
    classification_result: ClassificationResult,
    segmentation_summary: str,
) -> str:
    return f"""
You are assisting a CSDS312 project called "Brain Tumor Classification and Segmentation Using Image Processing."

Important rules:
1. Treat the project classification output as the primary tumor-type evidence.
2. Only use the project-supported labels: glioma, meningioma, pituitary, no_tumor.
3. Do not replace those labels with a more specific diagnosis that the classifier did not predict.
4. Be explicit that this is a project-model interpretation, not a definitive clinical diagnosis.
5. If segmentation is unavailable, say so clearly instead of inventing segmentation findings.

User question:
{question}

Project classification findings:
- Predicted label: {classification_result.predicted_label}
- Confidence: {classification_result.confidence:.2%}
- Full probability ranking: {format_probability_summary(classification_result)}

Project segmentation findings:
{segmentation_summary}

Write a concise answer that:
- answers the user's question directly,
- cites the project prediction first,
- briefly explains how the classification and segmentation parts of the project relate to this answer.
""".strip()


def build_fallback_response(
    question: str,
    classification_result: ClassificationResult,
    segmentation_summary: str,
) -> str:
    return (
        f'For the question "{question}", our classification model predicts '
        f'"{classification_result.predicted_label}" with '
        f'{classification_result.confidence:.2%} confidence. '
        f"The ranked probabilities are {format_probability_summary(classification_result)}. "
        f"Segmentation status: {segmentation_summary} "
        "This should be treated as a project-model interpretation rather than a definitive clinical diagnosis."
    )


def generate_project_answer(
    image_path: Path,
    question: str,
    classifier: BrainTumorClassifier,
    segmentation_summary: str,
    use_vlm: bool,
    model_name: str,
    max_new_tokens: int,
) -> str:
    classification_result = classifier.classify(image_path)

    llm_response = None
    if use_vlm:
        prompt = build_project_prompt(question, classification_result, segmentation_summary)
        try:
            llm_client = VisionLLMClient(model_name=model_name)
            llm_response = llm_client.ask_image(
                image_path=image_path,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:
            llm_response = (
                "The vision-language model could not be loaded, so a classifier-only summary "
                f"is shown instead. Details: {exc}"
            )

    if not llm_response:
        llm_response = build_fallback_response(question, classification_result, segmentation_summary)

    return "\n".join(
        [
            "Project findings:",
            f"- Predicted class: {classification_result.predicted_label} ({classification_result.confidence:.2%})",
            f"- Class probabilities: {format_probability_summary(classification_result)}",
            f"- Segmentation summary: {segmentation_summary}",
            "",
            "LLM interpretation:",
            llm_response,
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Brain tumor project assistant that combines CNN classification results with a vision LLM.",
    )
    parser.add_argument("--image", help="Path to the image file to analyze.")
    parser.add_argument("--question", help="Question to ask about the image.")
    parser.add_argument(
        "--classifier-checkpoint",
        default=str(DEFAULT_CLASSIFIER_CHECKPOINT),
        help="Path to the trained classification checkpoint.",
    )
    parser.add_argument(
        "--segmentation-summary",
        help="Optional path to a text or JSON file containing segmentation results.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_VLM_NAME,
        help="Vision-language model name on Hugging Face.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate for the LLM response.",
    )
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="Skip the vision-language model and print only the project-grounded summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_image_path = args.image or input("Enter image file name: ").strip()
    question = args.question or input("Enter your question: ").strip()
    question = question or "What does this image suggest in the context of our project?"

    image_path = resolve_input_path(raw_image_path)
    checkpoint_path = resolve_input_path(args.classifier_checkpoint)
    segmentation_summary = load_segmentation_summary(args.segmentation_summary)

    classifier = BrainTumorClassifier(checkpoint_path=checkpoint_path)
    answer = generate_project_answer(
        image_path=image_path,
        question=question,
        classifier=classifier,
        segmentation_summary=segmentation_summary,
        use_vlm=not args.skip_vlm,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
    )

    print("\nModel answer:")
    print(answer)


if __name__ == "__main__":
    main()
