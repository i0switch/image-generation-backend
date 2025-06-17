# app/main.py

import torch
import numpy as np
import cv2
import base64
import io
from PIL import Image
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from diffusers import StableDiffusionPipeline, ControlNetModel, DDIMScheduler
from insightface.app import FaceAnalysis

from models import ImageGenerationRequest

# グローバル変数としてモデルを保持
models = {}

# FastAPIのライフサイクルイベント：起動時にモデルをロード
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")
    # デバイスの自動選択（Hugging Face SpacesのGPUを利用）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32

    # 1. 顔分析モデル (InsightFace)
    # 人物の顔の特徴を抽出するために使用します
    face_app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    
    # 2. ControlNetモデル (InstantID)
    # 顔の構造を維持するために使用します
    controlnet = ControlNetModel.from_pretrained(
        "InstantID/ControlNetModel", 
        torch_dtype=dtype
    )
    
    # 3. ベースとなる画像生成モデル (RealBeautyMix)
    # ControlNetを組み込んでパイプラインを初期化します
    pipe = StableDiffusionPipeline.from_pretrained(
        "SG161222/RealBeautyMix",
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None, # 安全チェックを無効化
        use_safetensors=True,
    ).to(device)

    # 4. IP-Adapter (InstantID)
    # 顔のアイデンティティ（誰であるか）を画像に反映させます
    pipe.load_ip_adapter("InstantID/ip-adapter", subfolder="models", weight_name="ip-adapter.bin")
    
    # グローバル変数にロードしたモデルを格納
    models['face_app'] = face_app
    models['pipe'] = pipe
    print("Models loaded successfully!")
    yield
    # クリーンアップ
    models.clear()
    print("Models cleared.")

# FastAPIアプリのインスタンス化
app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"status": "Backend is running and models are ready"}

@app.post("/generate")
async def generate_image(request: ImageGenerationRequest):
    try:
        # Base64から画像をデコード
        face_image_bytes = base64.b64decode(request.face_image_b64)
        face_image = Image.open(io.BytesIO(face_image_bytes)).convert("RGB")

        # 顔分析を実行して特徴量を取得
        face_info = models['face_app'].get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
        if not face_info:
            raise HTTPException(status_code=400, detail="No face detected in the provided image.")
        
        face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1] # 最大の顔を選択
        face_emb = face_info['embedding']

        # IP-Adapterのスケールを設定
        models['pipe'].set_ip_adapter_scale(request.ip_adapter_scale)

        # 画像生成パイプラインを実行
        images = models['pipe'](
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            image_embeds=[face_emb],
            image=face_image, # ControlNet用の入力画像
            controlnet_conditioning_scale=request.ip_adapter_scale,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
        ).images

        # 生成された画像をBase64にエンコードして返す
        generated_image = images[0]
        buffered = io.BytesIO()
        generated_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"image_b64": img_str}

    except Exception as e:
        print(f"Error during image generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))