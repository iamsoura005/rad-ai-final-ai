import base64
import io
import json
import mimetypes
import os
from typing import Callable, Dict, Generator, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from groq import Groq
from pydantic import BaseModel, Field

try:
    from .region_parser import extract_region
except ImportError:
    from region_parser import extract_region

HEATMAP_IMPORT_ERROR = "Heatmap feature has been disabled."
HEATMAP_AVAILABLE = False
GRADCAM_ENGINE = None

load_dotenv()

INVALID_KEY_VALUES = {"", "your_actual_key", "your_key_here", "YOUR_GROQ_API_KEY"}
DEFAULT_ANALYZE_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DEFAULT_TEXT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

SYSTEM_PROMPT = (
    "You are an expert Radiology AI Assistant with deep knowledge in medical imaging analysis, trained to interpret X-rays, CT scans, MRI, and ultrasound images.\n\n"
    "Your role is to assist doctors by providing accurate, structured, and clinically relevant analysis of medical images.\n\n"
    "IMPORTANT RULES:\n"
    "- You MUST behave like a professional radiologist.\n"
    "- You MUST NOT guess randomly. If uncertain, say \"Insufficient evidence\".\n"
    "- You MUST clearly distinguish between observation and diagnosis.\n"
    "- You MUST keep explanations simple, clear, and medically correct.\n"
    "- You MUST NOT provide treatment or medication advice.\n"
    "- You MUST include confidence level in every analysis.\n\n"
    "INPUT:\n"
    "You will receive:\n"
    "1. A medical image (X-ray, MRI, CT, etc.)\n"
    "2. Optional model predictions or extracted features\n\n"
    "OUTPUT FORMAT (STRICT):\n\n"
    "1. Image Type:\n"
    "   - (X-ray / MRI / CT Scan / etc.)\n\n"
    "2. Region Analyzed:\n"
    "   - (e.g., Chest, Brain, Spine, Abdomen)\n\n"
    "3. Key Observations:\n"
    "   - List visible abnormalities (e.g., lesions, fractures, opacities)\n"
    "   - Mention normal findings if relevant\n\n"
    "4. Possible Condition(s):\n"
    "   - Primary suspected condition\n"
    "   - Secondary possibilities (if any)\n\n"
    "5. Severity Assessment:\n"
    "   - Mild / Moderate / Severe / Critical\n"
    "   - Severity Score: (0–10 scale)\n\n"
    "6. Clinical Explanation (Simple Language):\n"
    "   - Explain what is happening in easy terms\n\n"
    "7. Confidence Score:\n"
    "   - (0–100%)\n\n"
    "8. Recommendation:\n"
    "   - Suggest if further tests or expert consultation is needed (NO treatment advice)\n\n"
    "9. Detected Region:\n"
    "   - Include one final line exactly as: Detected Region: <region name or none>\n\n"
    "10. Heatmap Locations:\n"
    "   - Add one final line exactly as: Heatmap Locations: <comma separated locations or none>\n"
    "   - Preferred location terms by scan type:\n"
    "     * Chest: left lower lobe, right lower lobe, left upper lobe, right upper lobe, left lung, right lung, bilateral, heart/cardiac, mediastinum\n"
    "     * Bone femur: femoral head, femoral neck, hip fracture, mid-shaft, transverse fracture, diaphyseal, distal femur, supracondylar, knee, comminuted\n"
    "     * Bone wrist: distal radius, colles fracture, smith fracture, radial shaft, ulna, ulnar fracture\n"
    "     * Brain: left temporal lobe, right temporal lobe, left frontal lobe, right frontal lobe, left parietal lobe, right parietal lobe, midline shift, bilateral, cerebellum\n\n"
    "SAFETY:\n"
    "- Always include: \"This is an AI-assisted analysis and not a medical diagnosis.\"\n"
    "- Avoid definitive claims unless very clear.\n"
    "- If image quality is poor, mention it.\n\n"
    "STYLE:\n"
    "- Professional, concise, and structured\n"
    "- Easy English (important)\n"
    "- No unnecessary jargon\n\n"
    "GOAL:\n"
    "Provide highly reliable, explainable radiology insights that can assist in early detection and understanding of medical conditions."
)

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "application/dicom",
    "application/dicom+json",
}

SYMPTOMS_SYSTEM_PROMPT = (
    "You are an expert clinical symptom triage assistant. Analyze user symptoms and respond in this strict structure:\n\n"
    "1. Probable Conditions:\n"
    "- Condition Name (confidence %) : short reason\n"
    "- Include 2 to 5 possibilities\n\n"
    "2. Urgency Level:\n"
    "- Home Care / See a Doctor / Emergency\n"
    "- Give one-line urgency reason\n\n"
    "3. Possible Causes:\n"
    "- Bullet list of likely causes\n\n"
    "4. What to Do Next:\n"
    "- Numbered action steps (3 to 6)\n\n"
    "5. Safety Note:\n"
    "- Include clear disclaimer that this is AI guidance and not a diagnosis.\n"
    "Use simple language and avoid definitive diagnosis claims."
)

MEDICINE_SYSTEM_PROMPT = (
    "You are an expert pharmacist assistant. Provide medicine information in strict structure:\n\n"
    "1. Generic Name & Drug Class\n"
    "2. Uses & Indications\n"
    "3. Dosage & Administration\n"
    "4. Side Effects (common + serious)\n"
    "5. Contraindications & Warnings\n"
    "6. Drug Interactions\n"
    "7. Storage Instructions\n"
    "8. Safety Disclaimer\n\n"
    "Use concise medical language that non-experts can understand."
)

CHAT_SYSTEM_PROMPT = (
    "You are Dr. AI, a knowledgeable and compassionate AI medical assistant. "
    "You provide clear, evidence-based health information. You NEVER diagnose conditions definitively. "
    "You ALWAYS recommend consulting a real doctor for serious concerns. "
    "You explain medical concepts in simple language. "
    "You MUST include a disclaimer that you are an AI, not a real doctor."
)


class SymptomsRequest(BaseModel):
    symptoms: str = Field(min_length=3)
    age: int = Field(ge=0, le=130)
    gender: str = Field(min_length=1)
    duration: str = Field(min_length=1)


class MedicineRequest(BaseModel):
    medicine_name: str = Field(min_length=1)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(default_factory=list)

app = FastAPI(title="RadiologyAI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _infer_mime_type(upload: UploadFile) -> str:
    if upload.content_type:
        return upload.content_type

    guessed, _ = mimetypes.guess_type(upload.filename or "")
    return guessed or "application/octet-stream"


def _validate_mime_type(mime_type: str) -> None:
    if mime_type in ALLOWED_MIME_TYPES:
        return

    raise HTTPException(
        status_code=400,
        detail="Unsupported file format. Please upload JPG, PNG, WEBP, or DCM preview.",
    )


def _build_client() -> Groq:
    load_dotenv(override=True)
    groq_api_key = (os.getenv("GROQ_API_KEY") or "").strip()
    if groq_api_key in INVALID_KEY_VALUES:
        raise HTTPException(
            status_code=400,
            detail="GROQ_API_KEY is not set. Update backend/.env with your real Groq key.",
        )
    return Groq()


def _available_model_ids(client: Groq) -> List[str]:
    try:
        return [model.id for model in client.models.list().data]
    except Exception:
        return []


def _is_general_chat_model(model_id: str) -> bool:
    value = model_id.lower()
    blocked = ("guard", "whisper", "tts", "transcribe")
    return not any(tag in value for tag in blocked)


def _pick_model(client: Groq, purpose: str) -> str:
    available = _available_model_ids(client)

    if purpose == "analyze":
        env_model = (os.getenv("GROQ_ANALYZE_MODEL") or "").strip()
        preferred = [
            env_model,
            DEFAULT_ANALYZE_MODEL,
            "meta-llama/llama-4-maverick-17b-128e-instruct",
        ]
    else:
        env_model = (os.getenv("GROQ_TEXT_MODEL") or "").strip()
        preferred = [
            env_model,
            DEFAULT_TEXT_MODEL,
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ]

    preferred = [model_id for model_id in preferred if model_id]

    for model_id in preferred:
        if not available or model_id in available:
            return model_id

    for model_id in available:
        if "llama-4" in model_id.lower() and _is_general_chat_model(model_id):
            return model_id

    for model_id in available:
        if _is_general_chat_model(model_id):
            return model_id

    return preferred[0] if preferred else DEFAULT_TEXT_MODEL


def _stream_chat_completion(
    client: Groq,
    messages: list,
    model: str,
    fallback_model: str = "",
    post_payload_builder: Optional[Callable[[str], Optional[Dict]]] = None,
) -> Generator[str, None, None]:
    def _stream_with_model(model_id: str) -> Generator[str, None, str]:
        report_chunks: List[str] = []
        completion = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )

        for chunk in completion:
            delta = chunk.choices[0].delta
            chunk_text = getattr(delta, "content", None)
            if chunk_text:
                clean_chunk = chunk_text.replace("\r", "")
                report_chunks.append(clean_chunk)
                payload = json.dumps({"chunk": clean_chunk}, ensure_ascii=False)
                yield f"data: {payload}\n\n"

        return "".join(report_chunks)

    try:
        report_text = yield from _stream_with_model(model)

        if post_payload_builder:
            try:
                post_payload = post_payload_builder(report_text)
                if post_payload:
                    payload = json.dumps(post_payload, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
            except Exception:
                pass

        yield "data: [DONE]\n\n"
    except Exception as exc:
        error_text = str(exc).replace("\n", " ")

        can_retry = (
            fallback_model
            and fallback_model != model
            and (
                "does not exist" in error_text.lower()
                or "do not have access" in error_text.lower()
                or "not found" in error_text.lower()
            )
        )

        if can_retry:
            try:
                report_text = yield from _stream_with_model(fallback_model)

                if post_payload_builder:
                    try:
                        post_payload = post_payload_builder(report_text)
                        if post_payload:
                            payload = json.dumps(post_payload, ensure_ascii=False)
                            yield f"data: {payload}\n\n"
                    except Exception:
                        pass

                yield "data: [DONE]\n\n"
                return
            except Exception as fallback_exc:
                error_text = str(fallback_exc).replace("\n", " ")

        if "does not exist" in error_text.lower() or "do not have access" in error_text.lower():
            error_text = (
                "Configured Groq model is unavailable for this API key. "
                "Set GROQ_ANALYZE_MODEL or GROQ_TEXT_MODEL to an accessible model in deployment env."
            )

        error_message = error_text
        payload = json.dumps({"error": error_message}, ensure_ascii=False)
        yield f"data: {payload}\n\n"
        yield "data: [DONE]\n\n"


def _sse_response(stream: Generator[str, None, None]) -> StreamingResponse:
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze")
@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), context: str = Form(default="")) -> StreamingResponse:
    client = _build_client()
    model = _pick_model(client, "analyze")
    fallback_model = _pick_model(client, "text")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    mime_type = _infer_mime_type(file)
    _validate_mime_type(mime_type)

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    b64_string = base64.b64encode(file_bytes).decode("utf-8")
    image_data_url = f"data:{mime_type};base64,{b64_string}"

    user_text = "Please analyze this medical image."
    if context.strip():
        user_text = f"Patient context: {context.strip()}\n\nPlease analyze this medical image."

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]

    return _sse_response(
        _stream_chat_completion(
            client,
            messages,
            model,
            fallback_model,
            post_payload_builder=None,
        )
    )


@app.post("/symptoms")
@app.post("/api/symptoms")
async def analyze_symptoms(payload: SymptomsRequest) -> StreamingResponse:
    client = _build_client()
    model = _pick_model(client, "text")
    user_text = (
        f"Symptoms: {payload.symptoms}\n"
        f"Age: {payload.age}\n"
        f"Gender: {payload.gender}\n"
        f"Duration: {payload.duration}\n"
        "Provide a structured clinical triage style analysis."
    )
    messages = [
        {"role": "system", "content": SYMPTOMS_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    return _sse_response(_stream_chat_completion(client, messages, model))


@app.post("/medicine")
@app.post("/api/medicine")
async def medicine_info(payload: MedicineRequest) -> StreamingResponse:
    client = _build_client()
    model = _pick_model(client, "text")
    user_text = (
        f"Medicine name: {payload.medicine_name}\n"
        "Return complete medicine information in the requested structured format."
    )
    messages = [
        {"role": "system", "content": MEDICINE_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    return _sse_response(_stream_chat_completion(client, messages, model))


@app.post("/chat")
@app.post("/api/chat")
async def health_chat(payload: ChatRequest) -> StreamingResponse:
    client = _build_client()
    model = _pick_model(client, "text")
    messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
    for message in payload.messages:
        role = message.role.strip().lower()
        if role not in {"user", "assistant", "system"}:
            continue
        messages.append({"role": role, "content": message.content})

    if len(messages) == 1:
        messages.append(
            {
                "role": "user",
                "content": "Hello Dr. AI. Please introduce yourself and ask how you can help.",
            }
        )

    return _sse_response(_stream_chat_completion(client, messages, model))
