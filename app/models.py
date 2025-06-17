from pydantic import BaseModel
from typing import Optional

class ImageGenerationRequest(BaseModel):
    face_image_b64: str  # ユーザーが指定する人物の顔写真 (Base64)
    prompt: str
    negative_prompt: Optional[str] = "ugly, disfigured, low quality, blurry"
    num_inference_steps: Optional[int] = 30
    guidance_scale: Optional[float] = 7.5