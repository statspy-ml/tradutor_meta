from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText

# Initialize FastAPI app
app = FastAPI()

# Initialize the model and processor
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2ForTextToText.from_pretrained("facebook/seamless-m4t-v2-large")

class TranslationRequest(BaseModel):
    text: str
    source_language: str = "eng"
    target_language: str = "por"

@app.post("/translate/")
async def translate(request: TranslationRequest):
    try:
        text_inputs = processor(text=request.text, src_lang=request.source_language, return_tensors="pt")
        decoder_input_ids = model.generate(**text_inputs, tgt_lang=request.target_language)[0].tolist()
        translated_text = processor.decode(decoder_input_ids, skip_special_tokens=True)
        return {"translated_text": translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
