from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import os
from bit.model import BitModel
from bit.config import BITConfig
from bit.tokenizer import BITTokenizer
from bit.inference import BITInference

app = FastAPI(title="BIT API Server")

# Global state
config = BITConfig()
tokenizer = BITTokenizer(config)
model = None
inference = None

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95

class GenerateResponse(BaseModel):
    text: str

@app.on_event("startup")
async def startup_event():
    global model, inference
    if not os.path.exists(config.model_path):
        print(f"Model path {config.model_path} not found. Server will start but generation will fail.")
        return
        
    print(f"Loading model from {config.model_path}...")
    checkpoint = torch.load(config.model_path, map_location=config.device)
    model = BitModel(config).to(config.device)
    model.load_state_dict(checkpoint["model_state"])
    inference = BITInference(model, tokenizer, config)
    print("Model loaded successfully.")

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        text = inference.generate(
            request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p
        )
        return GenerateResponse(text=text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
