# app/models.py

from pydantic import BaseModel
from typing import Optional

class ImageGenerationRequest(BaseModel):
    face_image_b64: str
    prompt: str
    negative_prompt: Optional[str] = "(lowres, low quality, worst quality:1.2), (text:1.2), error, (missing fingers), (missing arms), (extra legs), (extra arms), (extra_fingers), (ugly:1.2), (duplicate), (morbid), (mutilated), (tranny:1.2), mutated hands, (poorly drawn hands), (poorly drawn face), (mutation:1.2), (deformed:1.2), (amputee:1.2), (bad anatomy), (bad proportions), (body contortions), (blurry), (fused fingers), (too many fingers), (uncoordinated body), (bad body), (bad eyes), (bad facial), (bad lighting), (bad background), (bad color), (bad perspective), (bad 2d), (bad 3d), (bad 4d), (bad hands)"
    num_inference_steps: Optional[int] = 30
    guidance_scale: Optional[float] = 5.0
    ip_adapter_scale: Optional[float] = 0.8