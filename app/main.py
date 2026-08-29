"""
main.py — FastAPI endpoint สำหรับคุยกับตัวละคร

จุดที่ต่างจาก chat API ทั่วไป: request เปิดให้ "ปิดกลไกป้องกันทีละชั้น" ได้
(use_prompt / use_stop_sequences / use_guard) เพื่อให้ tests/benchmark.py
วัดผลแบบ ablation ได้ว่าแต่ละชั้นช่วยลดปัญหาไปเท่าไหร่ ไม่ใช่วัดรวมแล้วเดาเอา

สำคัญ: ตัวตรวจจับ "คิดแทนผู้เล่น" จะทำงาน "ทุกครั้ง" ไม่ว่าจะเปิด use_guard หรือไม่
       ต่างกันแค่ว่าจะเอาผลไปตัดข้อความจริงไหม
       ทำแบบนี้เพื่อให้เงื่อนไขที่ปิด guard ยังมีตัวเลขให้นับได้
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # ไม่มี python-dotenv ก็ยังรันได้ ถ้าตั้ง env var เอง
    pass

from app import db, guards, llm

CHARACTERS_DIR = Path(__file__).parent / "characters"

# prompt ขั้นต่ำสุด ใช้ตอน use_prompt=False
#
# ทำไมไม่ใช้ "ไม่ส่ง system prompt เลย": เพราะแบบนั้นจะกลายเป็นการวัดว่า
# "เล่นบทบาทกับไม่เล่นบทบาท" ต่างกันแค่ไหน ซึ่งไม่ใช่สิ่งที่เราอยากรู้
# เราอยากรู้ว่า "กฎกันคิดแทนผู้เล่นและกฎความเด็ดขาด" ต่างหากที่ช่วยเท่าไหร่
# baseline จึงยังเล่นบทบาทอยู่ แค่ไม่มีกฎพวกนั้น
MINIMAL_PROMPT = (
    "คุณคือ {display_name} กำลังเล่นบทบาทสมมติกับผู้เล่นชื่อ {player_name}\n"
    "ฉาก: {scene}\n"
    "ตอบเป็นภาษาไทย ตอบเป็นบทสนทนาต่อเนื่องหนึ่งแบบ ไม่ต้องเสนอตัวเลือกให้เลือก"
)
# บรรทัดสุดท้ายเป็นเรื่องรูปแบบการตอบล้วนๆ ไม่ใช่กฎกันคิดแทนผู้เล่นหรือกฎความเด็ดขาด
# ต้องใส่เพราะตอนทดสอบครั้งแรกโมเดลตอบกลับมาเป็นเมนูตัวเลือกภาษาอังกฤษ
# ("2 (More immersive roleplay):") ซึ่งเอาไปวัดผลไม่ได้เลย
# การเทียบจึงยังยุติธรรม เพราะตัวแปรที่เราสนใจ (กฎสองข้อนั้น) ยังไม่อยู่ใน baseline


def load_character(character_id: str) -> dict:
    path = CHARACTERS_DIR / f"{character_id}.json"
    if not path.exists():
        available = sorted(p.stem for p in CHARACTERS_DIR.glob("*.json"))
        raise HTTPException(
            status_code=404,
            detail=f"ไม่พบตัวละคร '{character_id}' — ที่มีอยู่: {available}",
        )
    return json.loads(path.read_text(encoding="utf-8"))


def build_system_prompt(character: dict, player_name: str, full: bool = True) -> str:
    fields = {
        "display_name": character.get("display_name", ""),
        "scene": character.get("scene", ""),
        "persona": character.get("persona", ""),
        "goal_this_scene": character.get("goal_this_scene", ""),
        "wont_yield_on": character.get("wont_yield_on", ""),
        "idle_action": character.get("idle_action", ""),
        "player_name": player_name,
    }
    template = "\n".join(character["system_prompt_lines"]) if full else MINIMAL_PROMPT
    return template.format(**fields)


class ChatRequest(BaseModel):
    session_id: str
    player_name: str
    message: str
    character_id: str = "salesman"

    # สวิตช์สำหรับ ablation — ใช้งานปกติปล่อยเป็น True ทั้งหมด
    use_prompt: bool = True           # ชั้นที่ 3: กฎในบท system prompt
    use_stop_sequences: bool = True   # ชั้นที่ 1: หยุดที่ระดับ API
    use_guard: bool = True            # ชั้นที่ 2: ตัวกรอง regex ขาออก
    use_input_guard: bool = True      # ปัญหาที่ 4: ตัวกรองขาเข้า กันผู้ใช้ยัดคำพูดใส่ปากบอท


class ChatResponse(BaseModel):
    reply: str
    raw_reply: str
    guard_triggered: bool     # ตรวจเจอว่าเขียนแทนผู้เล่น (ตรวจทุกครั้ง แม้ปิด guard)
    guard_applied: bool       # ได้ตัดข้อความจริงหรือเปล่า
    guard_emptied: bool       # ตัดแล้วไม่เหลืออะไร
    ends_with_question: bool
    has_action: bool
    truncated: bool = False   # คำตอบชนเพดาน token — เทิร์นนี้เอาไปนับผลไม่ได้
    finish_reason: str = ""
    input_guard_triggered: bool = False   # ผู้ใช้พยายามเขียนบทให้ตัวละคร
    input_guard_rule: str = ""
    llm_retries: int = 0      # ยิงซ้ำกี่ครั้งเพราะโมเดลตอบว่าง


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print(f"[startup] {db.init_db()}")
    except Exception as e:  # ไม่ให้เซิร์ฟเวอร์ตายเพราะ DB ตั้งค่าไม่ครบ
        print(f"[startup][warn] ต่อฐานข้อมูลไม่ได้: {e}")
    yield


app = FastAPI(title="Roleplay Agent API", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": llm.get_model_name(),
        "storage": "postgres" if db.USING_POSTGRES else "memory",
        "characters": sorted(p.stem for p in CHARACTERS_DIR.glob("*.json")),
    }


@app.post("/reset")
def reset(session_id: str):
    db.reset_session(session_id)
    return {"status": "reset", "session_id": session_id}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    character = load_character(req.character_id)
    system_prompt = build_system_prompt(character, req.player_name, full=req.use_prompt)

    history = db.get_history(req.session_id)
    stop_sequences = guards.get_stop_sequences(req.player_name) if req.use_stop_sequences else None

    # ตรวจขาเข้า: ผู้ใช้พยายามเขียนบทให้ตัวละครหรือเปล่า
    # ตรวจทุกครั้งเหมือนฝั่งขาออก ต่างกันแค่จะแนบหมายเหตุไปให้โมเดลไหม
    character_name = character.get("display_name", "")
    imp = guards.detect_impersonation(req.message, character_name)

    message_to_model = req.message
    if imp.triggered and req.use_input_guard:
        message_to_model = req.message + guards.build_impersonation_note(character_name)

    try:
        llm_result = llm.generate_response(
            system_prompt=system_prompt,
            history=history,
            user_message=message_to_model,
            stop_sequences=stop_sequences,
        )
        raw_reply = llm_result.text
    except RuntimeError as e:
        # ตั้งค่าไม่ครบ เช่นไม่มี API key — ลองใหม่ไปก็ไม่หาย
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # error จากฝั่ง Gemini — ส่ง status ที่สื่อความหมายกลับไป
        # เพื่อให้ tests/benchmark.py แยกออกว่าอันไหนควรรอแล้วลองใหม่
        raise HTTPException(
            status_code=llm.classify_error(e),
            detail=f"{type(e).__name__}: {e}",
        )

    # ตรวจทุกครั้ง แต่ตัดเฉพาะตอนเปิด guard
    result = guards.detect_player_narration(raw_reply, req.player_name)
    reply = result.text if req.use_guard else raw_reply

    # บันทึกข้อความต้นฉบับของผู้ใช้ ไม่ใช่ตัวที่แนบหมายเหตุ
    # ไม่งั้นหมายเหตุจะค้างอยู่ในประวัติแล้วถูกส่งซ้ำทุกเทิร์นถัดไป
    db.log_turn(req.session_id, "player", req.message)
    db.log_turn(req.session_id, "character", reply)

    return ChatResponse(
        reply=reply,
        raw_reply=raw_reply,
        guard_triggered=result.triggered,
        guard_applied=req.use_guard and result.triggered and not result.emptied,
        guard_emptied=result.emptied,
        ends_with_question=guards.ends_with_question(reply),
        has_action=guards.has_action(reply),
        truncated=llm_result.truncated,
        finish_reason=llm_result.finish_reason,
        input_guard_triggered=imp.triggered,
        input_guard_rule=imp.rule,
        llm_retries=llm_result.retries,
    )
