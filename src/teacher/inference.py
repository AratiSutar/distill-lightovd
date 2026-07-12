"""
Teacher model wrapper: OWL-ViT zero-shot object detection.

Used to generate pseudo-labels for training the lightweight student model.
"""

import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection


class OwlViTTeacher:
    """Wraps OWL-ViT for zero-shot, text-prompted object detection."""

    def __init__(
        self, model_id: str = "google/owlvit-base-patch32", device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = OwlViTProcessor.from_pretrained(model_id)
        self.model = OwlViTForObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def detect(
        self,
        image: Image.Image,
        text_queries: list[str],
        threshold: float = 0.1,
    ) -> dict:
        """
        Run zero-shot detection on a single image.

        Args:
            image: PIL Image (RGB)
            text_queries: list of text prompts, e.g. ["a photo of a cat"]
            threshold: confidence threshold for filtering detections

        Returns:
            dict with keys: boxes (list[list[float]]), scores (list[float]), labels (list[str])
        """
        inputs = self.processor(
            text=text_queries, images=image, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)

        results = self.processor.image_processor.post_process_object_detection(
            outputs=outputs, threshold=threshold, target_sizes=target_sizes
        )[0]

        boxes = results["boxes"].tolist()
        scores = results["scores"].tolist()
        labels = [text_queries[i] for i in results["labels"].tolist()]

        return {"boxes": boxes, "scores": scores, "labels": labels}
