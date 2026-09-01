"""
llm.py — เรียก LLM ผ่าน Google Gen AI SDK

หมายเหตุเรื่อง SDK:
    แพ็กเกจเก่า `google-generativeai` ถูก Google ประกาศเลิกใช้แล้ว
    (repo ทางการเปลี่ยนชื่อเป็น deprecated-generative-ai-python)
    ไฟล์นี้ใช้ตัวใหม่คือ `google-genai` ซึ่งเรียกผ่าน genai.Client()

การสลับไปใช้ผู้ให้บริการเจ้าอื่น (เช่น Anthropic) แก้แค่ฟังก์ชัน generate_response()
ฟังก์ชันเดียว ส่วนที่เหลือของระบบไม่ต้องแตะ
"""

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from google import genai
from google.genai import types

# รุ่นเริ่มต้น ตั้งทับได้ด้วย env var GEMINI_MODEL
#
# ทำไมถึงเลือกรุ่น Lite เป็นค่าเริ่มต้น ทั้งที่มีรุ่นใหญ่กว่าให้ใช้:
#   free tier ให้โควตาต่างกันคนละโลก
#       gemini-3.5-flash        RPM 5    RPD 20     <- รันชุดทดสอบไม่จบใน 1 วัน
#       gemini-3.5-flash-lite   RPM 15   RPD 500    <- รันได้สบายๆ
#   และงานสวมบทบาทไม่ต้องการการให้เหตุผลหลายขั้นแบบที่รุ่นใหญ่เก่งกว่า
#   ถ้ากฎกับตัวกรองทำให้รุ่นถูกที่สุดใช้งานได้ นั่นคือผลลัพธ์ที่มีค่ากว่าในทางปฏิบัติ
DEFAULT_MODEL = "gemini-3.5-flash-lite"

# ------------------------------------------------------------------------
# ทำไม max_output_tokens ถึงต้องสูงขนาดนี้ทั้งที่คำตอบยาวแค่ 2-4 ประโยค
#
# Gemini รุ่น 3 เป็นโมเดลสายคิดก่อนตอบ (thinking) และ thinking token
# ถูกนับรวมอยู่ในงบเดียวกับ output token
#
# ตอนแรกตั้งไว้ 400 ผลคือคำตอบถูกตัดกลางประโยคทุกอัน และยิ่งประวัติบทสนทนายาวขึ้น
# คำตอบยิ่งสั้นลง (40 ตัวอักษร -> 30 -> 22) เพราะ thinking กินงบไปหมด
#
# ที่อันตรายคือมัน "ไม่ error" — benchmark รันจนจบสวยงามแล้วรายงาน 0/20 ทุกคอลัมน์
# ซึ่งดูเหมือนผลลัพธ์ที่ดีมาก ทั้งที่จริงคือไม่มีข้อความให้วัดเลย
# จึงเพิ่มการเช็ค finish_reason ไว้ด้วย ดูใน LLMResult ข้างล่าง
# ------------------------------------------------------------------------
DEFAULT_MAX_OUTPUT_TOKENS = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "2048"))

# ลดโหมดคิดลงให้เหลือน้อยที่สุดเท่าที่โมเดลรองรับ
# งานสวมบทบาทไม่ต้องการการให้เหตุผลหลายขั้น และการปิดไว้ทำให้ตอบเร็วกับถูกกว่า
THINKING_LEVEL = os.environ.get("GEMINI_THINKING_LEVEL", "low")

# จำนวนเทิร์นล่าสุดที่ส่งเข้า context = ความจำระยะสั้น (ชั้นที่ 1 ของระบบความจำ 3 ชั้น)
SHORT_TERM_WINDOW = 15

_client: Optional["genai.Client"] = None


def get_client() -> "genai.Client":
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ไม่พบ GEMINI_API_KEY — ตั้งค่าใน .env ก่อน (ดูตัวอย่างใน .env.example)"
            )
        _client = genai.Client(api_key=api_key)
    return _client


def get_model_name() -> str:
    return os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL


def classify_error(exc: BaseException) -> int:
    """
    แปลง error จาก SDK เป็น HTTP status เพื่อให้ฝั่งที่เรียกตัดสินใจได้ว่าควรลองใหม่ไหม

    ที่ต้องมีฟังก์ชันนี้: ตอนทดสอบจริงเจอ Gemini ตอบ 503 UNAVAILABLE
    (โมเดลฝั่ง Google คนใช้เยอะเกิน) ซึ่งคนละเรื่องกับ 429 ที่เป็นโควตาเราเต็ม
    ถ้าไม่แยกสองอันนี้ ฝั่ง benchmark จะเดาไม่ถูกว่าควรรอแล้วลองใหม่ หรือควรยอมแพ้
    """
    text = f"{type(exc).__name__} {exc}".upper()
    if "RESOURCE_EXHAUSTED" in text or "429" in text or "RATE LIMIT" in text or "QUOTA" in text:
        return 429  # โควตาเราเต็ม — รอแล้วลองใหม่ได้
    if "UNAVAILABLE" in text or "503" in text or "OVERLOAD" in text:
        return 503  # ฝั่ง Google ล้น — รอแล้วลองใหม่ได้เหมือนกัน แต่คนละสาเหตุ
    if "DEADLINE" in text or "TIMEOUT" in text or "504" in text:
        return 504
    return 502


def build_contents(history: List[Dict[str, str]], user_message: str) -> List[dict]:
    """
    แปลงประวัติบทสนทนาเป็นรูปแบบที่ SDK รับ

    ส่งเป็นบทสนทนาจริง (role user/model สลับกัน) ไม่ใช่ยัดทุกอย่างเป็นข้อความก้อนเดียว
    เพราะโมเดลแยกออกชัดกว่าว่าอันไหนคำพูดของใคร ซึ่งช่วยลดการเผลอเขียนแทนผู้เล่นได้อีกทาง
    """
    contents: List[dict] = []
    for turn in history[-SHORT_TERM_WINDOW:]:
        role = "user" if turn.get("role") == "player" else "model"
        contents.append({"role": role, "parts": [{"text": turn.get("content", "")}]})
    contents.append({"role": "user", "parts": [{"text": user_message}]})
    return contents


# จำนวนครั้งที่ยิงซ้ำเมื่อโมเดลตอบว่าง
#
# ตอนวัดผลจริงเจอว่า 4 ใน 80 คำตอบ (5%) กลับมาเป็นข้อความว่าง
# โดย finish_reason เป็น MALFORMED_RESPONSE ไม่ใช่ MAX_TOKENS
# แปลว่าโมเดลตอบเสียเอง ไม่ได้ถูกตัดเพราะเพดาน token
#
# เป็นอาการที่พบในโมเดลรุ่นเล็ก และเกิดกระจายทั่วทุกเงื่อนไข ไม่เกาะกลุ่ม
# ถ้าปล่อยผ่าน ผู้ใช้ 1 ใน 20 คนจะเจอบอทเงียบใส่ ซึ่งใช้งานจริงไม่ได้
# จึงตรวจแล้วยิงใหม่ที่ชั้นนี้เลย ไม่ใช่โยนความว่างเปล่าออกไปให้ปลายทางรับมือเอง
EMPTY_RESPONSE_RETRIES = 2


@dataclass
class LLMResult:
    text: str
    finish_reason: str
    truncated: bool     # คำตอบถูกตัดเพราะชนเพดาน token — ข้อมูลเทิร์นนี้เชื่อถือไม่ได้
    retries: int = 0    # ยิงซ้ำไปกี่ครั้งเพราะโมเดลตอบว่าง


def _build_config(config_kwargs: dict) -> "types.GenerateContentConfig":
    """
    สร้าง config โดยพยายามลดโหมดคิดก่อน

    ชื่อพารามิเตอร์ของการคุม thinking ต่างกันไปตามรุ่นโมเดลและเวอร์ชัน SDK
    (บางที่ใช้ thinking_config บางที่ใช้ thinking_level ตรงๆ)
    จึงลองไล่ทีละแบบแล้วถอยกลับไปใช้ config ธรรมดาถ้าไม่มีแบบไหนใช้ได้
    ดีกว่าฮาร์ดโค้ดแบบเดียวแล้วพังเมื่อ SDK อัปเดต
    """
    if THINKING_LEVEL and THINKING_LEVEL.lower() not in ("", "off", "default"):
        attempts = [
            lambda: types.GenerateContentConfig(
                **config_kwargs,
                thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
            ),
            lambda: types.GenerateContentConfig(**config_kwargs, thinking_level=THINKING_LEVEL),
        ]
        for build in attempts:
            try:
                return build()
            except Exception:
                continue

    return types.GenerateContentConfig(**config_kwargs)


def generate_response(
    system_prompt: str,
    history: List[Dict[str, str]],
    user_message: str,
    stop_sequences: Optional[Sequence[str]] = None,
    temperature: float = 0.9,
    max_output_tokens: Optional[int] = None,
    model: Optional[str] = None,
    response_mime_type: Optional[str] = None,
    response_schema: Optional[object] = None,
) -> LLMResult:
    """
    response_mime_type / response_schema (เฟส 6): ให้ Gemini บังคับรูปแบบผลลัพธ์เป็น JSON
    ตาม schema ที่ให้ (เช่น pydantic model) — ใช้เวลาต้องการคำตอบที่ parse ได้แน่นอน
    เช่น divergence.py ที่ต้องแยก "เปลี่ยนไหม" ออกจาก "เปลี่ยนว่าอะไร" อย่างชัดเจน

    ทำไมต้องมี: ตอนแรก divergence.py สั่งด้วยข้อความล้วนๆ ว่า "ตอบคำเดียว" แต่โมเดลรุ่น lite
    ชอบเขียนคำอธิบายนำหน้าคำตอบแม้สั่งห้ามแล้ว (พบจริงตอนทดสอบ 31 ส.ค. 2569 — ตอบว่า
    "...เนื้อเรื่องจึงยังดำเนินตามต้นฉบับ\\n\\nไม่เปลี่ยน" ทำให้โค้ดที่เช็คแค่ขึ้นต้นด้วยคำว่า
    "ไม่เปลี่ยน" อ่านผิดว่ามีการเปลี่ยนแปลง) response_schema บังคับที่ระดับ API ไม่ใช่แค่ขอในข้อความ
    เลยไม่มีทางมีคำอธิบายปนมาได้อีก
    """
    client = get_client()

    config_kwargs = {
        "system_instruction": system_prompt,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS,
    }
    # ชั้นที่ 1 ของการกันคิดแทนผู้เล่น — ส่งไปกับ request ตรงๆ
    if stop_sequences:
        config_kwargs["stop_sequences"] = list(stop_sequences)
    if response_mime_type:
        config_kwargs["response_mime_type"] = response_mime_type
    if response_schema is not None:
        config_kwargs["response_schema"] = response_schema

    model_name = model or get_model_name()
    contents = build_contents(history, user_message)
    config = _build_config(config_kwargs)

    text, finish_reason, truncated = "", "", False

    for attempt in range(EMPTY_RESPONSE_RETRIES + 1):
        response = client.models.generate_content(
            model=model_name, contents=contents, config=config
        )

        finish_reason = ""
        try:
            finish_reason = str(response.candidates[0].finish_reason or "")
        except (AttributeError, IndexError, TypeError):
            pass

        try:
            text = (response.text or "").strip()
        except (AttributeError, ValueError):
            # SDK บางเวอร์ชันโยน error แทนที่จะคืน None เมื่อไม่มีเนื้อหาให้อ่าน
            text = ""

        # ชนเพดาน token = คำตอบไม่จบประโยค เอาไปวัดผลไม่ได้
        # ต้องรู้ให้ได้ ไม่ใช่ปล่อยผ่านเงียบๆ แล้วได้ตัวเลขสวยที่ไม่มีความหมาย
        truncated = "MAX_TOKEN" in finish_reason.upper()

        # ได้ข้อความมาแล้ว หรือถูกตัดเพราะ token (คนละปัญหา ยิงใหม่ไม่ช่วย) -> จบ
        if text or truncated:
            return LLMResult(
                text=text,
                finish_reason=finish_reason,
                truncated=truncated,
                retries=attempt,
            )

        # ตอบว่าง — พักแล้วยิงใหม่
        if attempt < EMPTY_RESPONSE_RETRIES:
            time.sleep(1.5 * (attempt + 1))

    # ยิงครบแล้วยังว่าง คืนตามจริง ไม่แต่งเติม
    return LLMResult(
        text=text,
        finish_reason=finish_reason,
        truncated=truncated,
        retries=EMPTY_RESPONSE_RETRIES,
    )
