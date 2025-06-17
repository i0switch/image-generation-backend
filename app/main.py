import torch
import base64
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from diffusers import StableDiffusionPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
from PIL import Image
import io

# カスタムモジュール
from models import ImageGenerationRequest
from instantid import InstantID  # InstantIDのパイプライン処理を別ファイルにまとめることを推奨

# グローバル変数としてモデルを保持
models = {}

# FastAPIのライフサイクルイベント：起動時にモデルをロード
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")
    # デバイスの自動選択（Hugging Face SpacesのGPUを利用）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # InstantIDに必要なControlNetと顔分析モデルをロード
    face_adapter_path = "path/to/your/iAdapter/model" # 事前にダウンロードまたはHubから指定
    controlnet_path = "ant-design/ant-design" # InstantIDのControlNet
    
    controlnet = ControlNetModel.from_pretrained(
        controlnet_path, torch_dtype=torch.float16, use_safetensors=True
    ).to(device)
    
    # ベースとなる画像生成モデルをロード
    base_model_path = "SG161222/RealBeautyMix"
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)
    
    # InstantIDパイプラインをインスタンス化
    models["instantid"] = InstantID(pipe, face_adapter_path)
    print("Models loaded successfully!")
    yield
    # クリーンアップ（今回は不要）
    models.clear()

# FastAPIアプリのインスタンス化
app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"status": "Backend is running"}

@app.post("/generate")
async def generate_image(request: ImageGenerationRequest):
    try:
        # Base64から画像をデコード
        face_image_bytes = base64.b64decode(request.face_image_b64)
        face_image = Image.open(io.BytesIO(face_image_bytes)).convert("RGB")

        # InstantIDパイプラインを実行
        # ※InstantIDの具体的な呼び出し方は公式実装に準拠します
        #   以下は概念的なコードです
        generated_image = models["instantid"].generate(
            face_image=face_image,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
        )

        # 生成された画像をBase64にエンコードして返す
        buffered = io.BytesIO()
        generated_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"image_b64": img_str}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Image generation failed.")

# InstantIDの処理をまとめるヘルパークラス/ファイル (例: instantid.py)
# この部分はHugging FaceのInstantID公式実装を参考に、使いやすいようにラップします。
# class InstantID:
#     def __init__(self, pipe, face_adapter_path): ...
#     def generate(self, face_image, prompt, ...): ...