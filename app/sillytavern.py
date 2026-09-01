"""
sillytavern.py — ชั้นแปลภาษา (compatibility shim) ให้ SillyTavern คุยกับ backend นี้ได้ (เฟส 7)

ทำไมต้องมีไฟล์นี้แยกจาก /chat เดิม:
  SillyTavern (และ client มาตรฐานทั่วไปสำหรับ LLM) ไม่รู้จัก endpoint /chat ที่เราออกแบบเอง
  (character_id, story_session, player_name เป็น field เฉพาะของโปรเจกต์นี้เท่านั้น) มันคาดหวัง
  backend ที่พูดภาษา "OpenAI-compatible Chat Completions" (POST /v1/chat/completions) ซึ่งเป็น
  มาตรฐานโดยพฤตินัยที่ frontend สำหรับ LLM แทบทุกตัวรองรับ (เลือกใน SillyTavern ผ่านโหมด
  "Chat Completion" → API = "Custom (OpenAI-compatible)")

  ไฟล์นี้แค่แปลรูปแบบคำขอ/คำตอบไปเรียก run_chat() เดิมใน main.py — ไม่แตะ RAG / guards /
  divergence เลยสักบรรทัด เพราะ run_chat() คือไปป์ไลน์เดียวกับที่ tests/benchmark.py และ
  tests/test_phase6.py วัดผลไว้แล้วทั้งหมด (ดูเหตุผลการแยกฟังก์ชันใน main.py)

ทำไม hardcode character_id/story_session แทนให้ SillyTavern เลือกเอง (ตกลงกับผู้ใช้ไว้ 31 ส.ค. 2569):
  โปรเจกต์นี้ยังเป็นต้นแบบส่วนตัว ยังไม่ต้องรองรับหลายตัวละครพร้อมกันผ่าน ST — เพดานความซับซ้อนที่ต้อง
  แลกคือทำ mapping "SillyTavern character card" → "character_id ของเรา" ซึ่งไม่จำเป็นตอนนี้
  ตั้งค่าผ่าน env var แทน (ST_CHARACTER_ID, ST_STORY_SESSION_ID, ST_PLAYER_NAME, ...) เปลี่ยนตัวละคร/
  เซฟที่คุยด้วยได้โดยไม่ต้องแก้โค้ด

ทำไมไม่รองรับ streaming (stream=true) ในรอบแรก:
  SillyTavern ส่ง stream=true มาเป็นค่าเริ่มต้นถ้าผู้ใช้ไม่ปิดเอง การ stream จริงต้องส่ง token/chunk
  จาก Gemini ออกมาทีละส่วนแล้วห่อเป็น SSE ของ OpenAI เอง เพิ่มความซับซ้อนอีกชั้นที่ไม่จำเป็นแค่เพื่อ
  พิสูจน์ว่าเชื่อมกันได้ — รอบแรกนี้บังคับ non-streaming ก่อน (ผู้ใช้ต้องปิด "Streaming" ใน ST เอง)
  ถ้า stream=true ส่งมาจริง จะตอบ 400 พร้อมข้อความบอกวิธีแก้ตรงๆ แทนที่จะเงียบแล้วพังแบบเดาสาเหตุไม่ออก

ทำไมมี GET /v1/models:
  SillyTavern เรียก endpoint นี้ตอนกดเชื่อมต่อ/โหลดรายชื่อโมเดลในหน้าตั้งค่า ถ้าไม่มี endpoint นี้
  จะขึ้น error ตั้งแต่ขั้นเชื่อมต่อ ก่อนจะได้ลองแชทด้วยซ้ำ — คืนรายชื่อโมเดลปลอมตัวเดียวพอ เพราะเราไม่ได้
  ให้ ST เลือกโมเดลจริงอยู่แล้ว (โมเดลจริงคุมที่ app/llm.py ฝั่งเราเอง)

ทำไม auto-สร้างเซฟให้เลย ถ้ายังไม่มี (ต่างจาก /story/start เดิมที่ผู้เล่นต้องเรียกเอง):
  ผู้ใช้ SillyTavern ไม่รู้จัก /story/start และไม่ควรต้องไปเปิด /docs ก่อนจะเริ่มคุยได้ — เทิร์นแรกที่
  ยิงมาจาก ST จึงสร้างเซฟให้อัตโนมัติด้วยค่า default จาก env var ถ้ายังไม่มีเซฟชื่อนี้อยู่ก่อน
  (สอดคล้องกับแนวทาง "hardcode ให้เรียบง่ายก่อน" ที่ตกลงกันไว้)

  หมายเหตุ: บรรทัดนี้ทำงานแค่ "ตอนยังไม่มีเซฟ" เท่านั้น ถ้าเซฟ ST_STORY_SESSION_ID มีอยู่แล้ว (เช่น
  เคยคุยไปก่อนหน้านี้) การแก้ ST_PLAYER_CHARACTER_NAME/DESC ด้านล่างจะไม่มีผลย้อนหลังกับเซฟเดิมเลย
  ต้องเรียก POST /story/{session_id}/reset (ดู main.py) พร้อมระบุ player_character ใหม่ตรงๆ ในคำขอ
  ถ้าอยากเปลี่ยนคอนเซปต์ตัวละครของเซฟที่มีอยู่แล้ว

ทำไมตัวละครผู้เล่นเริ่มต้นเป็น "เสียงในหัว" ไม่มีร่างกาย (เปลี่ยนจากเดิม 1 ก.ย. 2569):
  ตกลงกับผู้ใช้ไว้ว่าตัวละครแบบมีร่างกายแยกจากไคลน์ทำให้เกิดปัญหา "ผู้เล่นแยกฉากจากตัวเอก" — ผู้บรรยาย
  ไม่มีบันทึกต้นฉบับของฉากไคลน์ให้อ่านเวลาผู้เล่นไปทำอย่างอื่น (RAG ค้นจากข้อความผู้เล่น ไม่ใช่จาก
  สถานะไคลน์) เสียงในหัวแก้ปัญหานี้ตรงๆ เพราะอยู่กับไคลน์ทุกฉากโดยไม่มีทางแยกทางกันได้เลย — กฎเรื่อง
  ขอบเขตความสามารถ (พูดได้อย่างเดียว, ใครรับรู้ได้บ้าง, กันสปอยล์แบบกลมกลืนเรื่อง) อยู่ในกฎข้อ 12-17
  ของ app/characters/narrator_lotm.json ทั้งหมด คำอธิบายตัวละครด้านล่างนี้แค่สรุปให้ตรงกัน
"""
import os
import time
import uuid
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

router = APIRouter()

# ---- ตั้งค่าคงที่ผ่าน env var — ดูเหตุผลที่ hardcode แทนให้ ST เลือกเองในหัวไฟล์ ----
ST_CHARACTER_ID = os.environ.get("ST_CHARACTER_ID", "narrator_lotm")
ST_STORY_SESSION_ID = os.environ.get("ST_STORY_SESSION_ID", "sillytavern_session")
ST_PLAYER_NAME = os.environ.get("ST_PLAYER_NAME", "Biru")
ST_PLAYER_CHARACTER_NAME = os.environ.get("ST_PLAYER_CHARACTER_NAME", "เสียงในหัว")
ST_PLAYER_CHARACTER_DESC = os.environ.get(
    "ST_PLAYER_CHARACTER_DESC",
    "ตัวตนพิเศษที่มีอยู่แค่ในความคิดของไคลน์เท่านั้น เป็นเหมือนเสียงในหัวที่ไคลน์คนเดียวรับรู้ได้ "
    "ไม่มีร่างกาย สื่อสารกับไคลน์ได้ทางความคิด (ไม่ต้องออกเสียงก็คุยกันได้) ทำได้แค่พูดคุย/แนะนำ "
    "และช่วยดึงสติไคลน์กลับมาเมื่อเขาเสียการควบคุมหรือหวาดกลัวเกินไปเท่านั้น ไม่สามารถควบคุมร่างกาย"
    "ไคลน์ ควบคุมพลังหมอกเทา หรือทำสิ่งอื่นใดในโลกทางกายภาพได้เลย ไม่ใช่เป้าหมายที่ถูกคุกคามหรือ"
    "โจมตีได้โดยตรง (ภัยทุกอย่างตกอยู่ที่ไคลน์เสมอ) รับรู้เหตุการณ์รอบตัวไคลน์ได้ตลอดเวลาแม้ไคลน์จะ"
    "หลับหรือหมดสติ เพราะเห็น/ได้ยินผ่านประสาทสัมผัสเดียวกับไคลน์เสมอ รู้ความจริงอยู่แล้วว่าไคลน์ "
    "(ไคลน์ โมเรตติ) แท้จริงคือจิตของโจวหมิงรุ่ยที่เข้ามาสิงร่างนี้",
)
ST_MODEL_NAME = "roleplay-agent"   # ชื่อที่โผล่ในดรอปดาวน์ของ ST เฉยๆ ไม่ผูกกับโมเดลจริงฝั่ง Gemini


class OAChatMessage(BaseModel):
    role: str
    content: str


class OAChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    # ฟิลด์อื่นที่ SillyTavern อาจส่งมา (temperature, max_tokens, presence_penalty, ...) รับได้แต่
    # ไม่ใช้ เพราะโปรเจกต์นี้คุมค่าพวกนี้เองอยู่แล้วใน app/llm.py ไม่อยากให้ ST override โดยไม่ตั้งใจ
    # แล้วทำให้ผลไม่ตรงกับที่วัดไว้ในเฟสก่อนๆ
    model: Optional[str] = None
    messages: List[OAChatMessage]
    stream: Optional[bool] = False


def _extract_last_user_message(messages: List[OAChatMessage]) -> str:
    """เอาแค่ข้อความล่าสุดที่ role='user' — history เดิมทั้งหมดที่ ST ส่งมาไม่ใช้เลย

    เพราะ backend นี้คุมประวัติบทสนทนาเองอยู่แล้วผ่าน db.get_history(session_id) (ดู run_chat
    ใน main.py) การเอา history จาก ST มาใช้ซ้ำจะซ้อนกับของเดิม แถมเสี่ยงพัง prompt เพราะ ST
    แนบ system prompt/character card ของตัวเองมาด้วยในรูปแบบที่ backend เราไม่ได้ออกแบบให้อ่าน
    """
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    raise HTTPException(status_code=400, detail="ไม่พบข้อความจาก role='user' ใน messages")


@router.get("/v1/models")
def list_models() -> dict:
    return {
        "object": "list",
        "data": [{
            "id": ST_MODEL_NAME,
            "object": "model",
            "created": 0,
            "owned_by": "roleplay-agent",
        }],
    }


@router.post("/v1/chat/completions")
def chat_completions(req: OAChatRequest) -> dict:
    if req.stream:
        raise HTTPException(
            status_code=400,
            detail=(
                "ยังไม่รองรับ streaming ในเวอร์ชันนี้ — ไปที่หน้าตั้งค่า API ของ SillyTavern "
                "แล้วปิดตัวเลือก 'Streaming' ก่อน (ดูวิธีตั้งค่าในเอกสารโปรเจกต์ เฟส 7)"
            ),
        )

    user_message = _extract_last_user_message(req.messages)

    # import แบบ deferred (อยู่ในฟังก์ชัน ไม่ใช่หัวไฟล์) เพื่อกัน circular import
    # เพราะ main.py ก็ import router จากไฟล์นี้กลับไปเหมือนกัน — ดูคำอธิบายเต็มใน main.py
    from app import story_state
    from app.main import ChatRequest, run_chat

    # เทิร์นแรกที่ยิงมาจาก ST อาจยังไม่มีเซฟ — สร้างให้อัตโนมัติด้วยค่า default จาก env var
    # (ผู้ใช้ ST ไม่ควรต้องไปเปิด /docs เรียก /story/start เองก่อน — ดูเหตุผลในหัวไฟล์)
    if not story_state.exists(ST_STORY_SESSION_ID):
        story_state.load_or_create(ST_STORY_SESSION_ID, {
            "name": ST_PLAYER_CHARACTER_NAME,
            "description": ST_PLAYER_CHARACTER_DESC,
        })

    chat_req = ChatRequest(
        session_id=ST_STORY_SESSION_ID,
        player_name=ST_PLAYER_NAME,
        message=user_message,
        character_id=ST_CHARACTER_ID,
        story_session=ST_STORY_SESSION_ID,
    )
    result = run_chat(chat_req)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or ST_MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result.reply},
            "finish_reason": "stop",
        }],
        # ไม่ได้นับ token จริง — ST ไม่ได้บังคับให้ตัวเลขนี้แม่น แค่ต้องมีฟิลด์ให้ครบตามสคีมา
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
