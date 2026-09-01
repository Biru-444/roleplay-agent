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
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # ไม่มี python-dotenv ก็ยังรันได้ ถ้าตั้ง env var เอง
    pass

from app import chapter_watch, db, divergence, guards, llm, rag, story_state
from app.sillytavern import router as sillytavern_router
# ทำไม import ตอนหัวไฟล์ได้ทั้งที่ sillytavern.py ก็ import จาก main.py กลับมาเหมือนกัน (ดูในไฟล์นั้น):
# sillytavern.py import ChatRequest/run_chat แบบ deferred (อยู่ในฟังก์ชัน ไม่ใช่หัวไฟล์) จึงไม่มี
# circular import ตอนโหลดโมดูล — พอถึงเวลาเรียกจริง main.py โหลดเสร็จสมบูรณ์แล้วเสมอ

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


def build_system_prompt(character: dict, player_name: str, full: bool = True,
                         extra_fields: Optional[dict] = None) -> str:
    fields = {
        "display_name": character.get("display_name", ""),
        "scene": character.get("scene", ""),
        "persona": character.get("persona", ""),
        "goal_this_scene": character.get("goal_this_scene", ""),
        "wont_yield_on": character.get("wont_yield_on", ""),
        "idle_action": character.get("idle_action", ""),
        "player_name": player_name,
        # ---- เฟส 6: ค่าเริ่มต้นเผื่อไม่มีเซฟผูกอยู่ (character ที่ไม่ใช่ narrator ไม่อ้างถึงอยู่แล้ว) ----
        "player_character_name": player_name,
        "player_character_description": "",
        "current_chapter": "",
    }
    if extra_fields:
        fields.update(extra_fields)

    # ---- แก้บั๊กที่พบระหว่างตรวจกฎ 12-17 ใหม่ (เฟส 7, 1 ก.ย. 2569) ----
    #
    # เนื้อหาในฟิลด์ persona/goal_this_scene/wont_yield_on/idle_action ของ narrator_lotm.json เอง
    # ใช้ {player_character_name} และ {current_chapter} เป็นตัวแปรซ้อนอยู่ข้างในด้วย แต่ template.format()
    # ชั้นนอกด้านล่างทำแค่รอบเดียว ไม่ไล่ format ซ้ำเข้าไปในค่าที่เพิ่งถูกแทนเข้าไปแล้ว (พฤติกรรมปกติของ
    # str.format ของ Python) ผลคือก่อนแก้ตรงนี้ Gemini เห็นข้อความ "{player_character_name}" และ
    # "{current_chapter}" เป็นตัวอักษรดิบๆ ปนอยู่ในพรอมต์จริง 5 จุด (เจอตอนตรวจสอบเองก่อนส่งงานเฟส 7
    # ไม่ใช่จากการทดสอบของผู้ใช้ — เช็คด้วย regex หาวงเล็บปีกกาที่เหลือค้างในพรอมต์สุดท้าย)
    # แก้โดย format ฟิลด์เนื้อหาพวกนี้ก่อนหนึ่งรอบด้วยค่าที่มีใน fields อยู่แล้ว ห่อ try/except กันกรณี
    # ตัวละครไฟล์อื่นในอนาคตมีวงเล็บปีกกาที่ไม่ได้ตั้งใจให้เป็นตัวแปร (ดีกว่าทำทั้งเทิร์นพัง)
    for key in ("scene", "persona", "goal_this_scene", "wont_yield_on", "idle_action"):
        try:
            fields[key] = fields[key].format(**fields)
        except (KeyError, IndexError):
            pass

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
    use_rag: bool = True              # เฟส 5: ค้นคลังนิยายมาแนบใน prompt

    # ค้นมากี่ชิ้น — ค่าเริ่มต้น 8 ไม่ใช่ 5
    #
    # ตอนวัด recall@5 เราจำกัดที่ 5 เพราะต้องมีเลขเดียวไว้เทียบระหว่างวิธี
    # แต่ตอนใช้งานจริงข้อจำกัดนั้นไม่มีเหตุผล ชิ้นละ ~1,200 ตัวอักษร
    # ส่ง 8 ชิ้นก็ยังห่างเพดาน context ของ Gemini มาก
    # และ recall@8 ย่อมสูงกว่า recall@5 เสมอ = โอกาสที่ชิ้นถูกอยู่ใน prompt มากขึ้น
    rag_k: int = 8

    # ---- เฟส 6: ระบบเนื้อเรื่องแยกสาขา (divergence ledger) ----
    #
    # ใส่ story_session เพื่อเปิดโหมดนี้เท่านั้น — ไม่ใส่ = พฤติกรรมเดิมทุกประการ
    # (tests/benchmark.py เรียก /chat โดยไม่รู้จักฟิลด์นี้อยู่แล้ว จึงไม่กระทบผลวัดเดิม)
    # เมื่อเปิด: ผูก RAG ให้ค้นได้แค่ chapter <= current_chapter ของเซฟ (กันสปอยล์)
    # และแนบ divergence_log ต่อท้าย prompt ให้เนื้อเรื่องจำการเปลี่ยนแปลงที่ผู้เล่นทำไว้ได้
    story_session: Optional[str] = None
    use_divergence_extraction: bool = True   # เรียก LLM รอบเล็กหลังตอบ เพื่อสกัดข้อเท็จจริงที่เปลี่ยนไป

    # เรียก LLM รอบเล็กหลังตอบ เพื่อประเมินว่าฉากนี้น่าจะจบตอนหรือยัง (เฟส 7) — แค่ "เตือน" เก็บลงเซฟ
    # ไม่เลื่อนตอนให้เอง ดูเหตุผลเต็มใน app/chapter_watch.py
    use_chapter_watch: bool = True


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

    # ---- เฟส 5: ข้อมูลสำหรับตรวจสอบว่า RAG ทำงานยังไงในเทิร์นนี้ ----
    # มีไว้เพื่อแยกให้ออกว่าเวลาคำตอบผิด มันผิดเพราะ "ค้นมาผิด" หรือ "ค้นถูกแต่ AI มั่ว"
    # ถ้าไม่เก็บสามค่านี้ เวลาดีบักจะเดาไม่ถูกเลยว่าต้องไปแก้ตรงไหน
    rag_used: bool = False
    rag_query: str = ""            # คำที่เอาไปค้นจริง (อาจไม่ใช่ข้อความผู้เล่นตรงๆ)
    rag_chapters: List[int] = []   # ตอนที่ค้นมาได้ เรียงตามอันดับ
    rag_ms: int = 0

    # ---- เฟส 6: ผลลัพธ์ของระบบเนื้อเรื่องแยกสาขา ----
    story_chapter: Optional[int] = None    # current_chapter ของเซฟ ณ เทิร์นนี้ (None ถ้าไม่ได้ผูกเซฟ)
    divergence_detected: bool = False      # เทิร์นนี้ทำให้เนื้อเรื่องเบี่ยงจากต้นฉบับไหม
    divergence_fact: str = ""              # ข้อเท็จจริงที่บันทึกเพิ่ม (ถ้า divergence_detected)

    # ---- เฟส 7: ตัวเตือนว่าฉากนี้น่าจะจบตอนหรือยัง (ดู app/chapter_watch.py) ----
    # หมายเหตุ: SillyTavern อ่านไม่เห็น field พวกนี้ (เห็นแค่ reply) — ผลนี้ถูกบันทึกลงเซฟด้วยเสมอ
    # (story_state.chapter_advance_suggested/reason) ให้หน้า /story/{id}/panel อ่านมาแสดงแทน
    chapter_advance_suggested: bool = False
    chapter_advance_reason: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print(f"[startup] {db.init_db()}")
    except Exception as e:  # ไม่ให้เซิร์ฟเวอร์ตายเพราะ DB ตั้งค่าไม่ครบ
        print(f"[startup][warn] ต่อฐานข้อมูลไม่ได้: {e}")

    # โหลด index + โมเดล embedding ครั้งเดียวตอนสตาร์ท
    # กินเวลา 30-60 วินาที แต่ทำที่นี่ครั้งเดียว ดีกว่าโหลดทุก request
    # (warmup ไม่ raise — ถ้า RAG ไม่พร้อม /chat เดิมยังทำงานได้ตามปกติ)
    if os.environ.get("SKIP_RAG_WARMUP") != "1":
        print("[startup] กำลังโหลดระบบค้นหาคลังนิยาย (30-60 วินาที) ...")
        print(f"[startup] {rag.warmup()}")
    else:
        print("[startup] ข้ามการโหลด RAG (SKIP_RAG_WARMUP=1)")
    yield


app = FastAPI(title="Roleplay Agent API", lifespan=lifespan)
app.include_router(sillytavern_router)   # เฟส 7: ชั้นแปลภาษาให้ SillyTavern คุยด้วยได้ (ดู app/sillytavern.py)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": llm.get_model_name(),
        "storage": "postgres" if db.USING_POSTGRES else "memory",
        "characters": sorted(p.stem for p in CHARACTERS_DIR.glob("*.json")),
        "rag": rag.status(),
    }


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    max_per_chapter: Optional[int] = 2   # กันไม่ให้ได้ 5 ชิ้นจากตอนเดียวกันหมด


class SearchHit(BaseModel):
    rank: int
    chapter: int
    chapter_title: str
    text: str


class SearchResponse(BaseModel):
    query: str
    hits: List[SearchHit]
    elapsed_ms: int      # ดูว่าค้นเร็วพอจะใส่ใน /chat ไหม


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """ค้นคลังนิยายอย่างเดียว ยังไม่เกี่ยวกับ LLM

    endpoint นี้มีไว้ตรวจว่า retrieval ทำงานถูกในสภาพแวดล้อมจริงของเซิร์ฟเวอร์
    ก่อนจะเอาไปต่อกับ /chat — เวลาผลออกมาแปลก จะได้แยกออกว่าปัญหาอยู่ที่
    การค้น หรืออยู่ที่ prompt
    """
    if not rag.is_ready():
        raise HTTPException(
            status_code=503,
            detail=f"ระบบค้นหายังไม่พร้อม: {rag.status()['error']}",
        )
    t0 = time.perf_counter()
    hits = rag.search(req.query, k=req.k, max_per_chapter=req.max_per_chapter)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return SearchResponse(
        query=req.query,
        hits=[
            SearchHit(
                rank=h["rank"],
                chapter=h["chapter"],
                chapter_title=h["chapter_title"],
                text=h["text"],
            )
            for h in hits
        ],
        elapsed_ms=elapsed_ms,
    )


@app.post("/reset")
def reset(session_id: str):
    db.reset_session(session_id)
    return {"status": "reset", "session_id": session_id}


# ---------------------------------------------------------------------------
# เฟส 6: endpoint จัดการเซฟของระบบเนื้อเรื่องแยกสาขา (divergence ledger)
# ---------------------------------------------------------------------------

class DivergenceFactOut(BaseModel):
    chapter: int
    fact: str
    turn_index: int
    created_at: float


class StoryStateResponse(BaseModel):
    session_id: str
    current_chapter: int
    player_character: dict
    divergence_log: List[DivergenceFactOut]
    chapter_advance_suggested: bool = False   # เฟส 7 — ดู app/chapter_watch.py
    chapter_advance_reason: str = ""


def _story_response(state: story_state.StoryState) -> StoryStateResponse:
    return StoryStateResponse(
        session_id=state.session_id,
        current_chapter=state.current_chapter,
        player_character=state.player_character,
        divergence_log=[DivergenceFactOut(**f.__dict__) for f in state.divergence_log],
        chapter_advance_suggested=state.chapter_advance_suggested,
        chapter_advance_reason=state.chapter_advance_reason,
    )


class StoryStartRequest(BaseModel):
    session_id: str
    player_character: dict = {}   # เช่น {"name": "เอ็ดริค ฮอลโลว์", "description": "..."}


@app.post("/story/start", response_model=StoryStateResponse)
def story_start(req: StoryStartRequest):
    """สร้างเซฟใหม่ — ตอนเริ่มต้นเสมอคือตอนที่ 1 ตามที่ตกลงกันไว้ (ไม่มีการเริ่มกลางเรื่อง)"""
    if story_state.exists(req.session_id):
        raise HTTPException(
            status_code=400,
            detail=f"เซฟ '{req.session_id}' มีอยู่แล้ว — ดูสถานะที่ GET /story/{req.session_id}",
        )
    state = story_state.load_or_create(req.session_id, req.player_character)
    return _story_response(state)


class StoryResetRequest(BaseModel):
    # ไม่ใส่ (ปล่อยว่าง) = คงตัวละครผู้เล่นเดิมไว้ แค่ล้างตอนที่/divergence_log/ประวัติบทสนทนา
    # ใส่ = เปลี่ยนตัวละครผู้เล่นไปพร้อมกันในคำสั่งเดียว (เช่นเปลี่ยนคอนเซปต์ตัวละครใหม่)
    player_character: dict = {}


@app.post("/story/{session_id}/reset", response_model=StoryStateResponse)
def story_reset(session_id: str, req: StoryResetRequest):
    """ล้างเซฟกลับไปตอนที่ 1 + ล้าง divergence_log + ล้างประวัติบทสนทนา ในคำสั่งเดียว (เฟส 7)

    ทำไมต้องมี endpoint นี้: ก่อนหน้านี้ไม่มีทางเริ่มทดสอบสถานการณ์ใหม่แบบสะอาดได้เลยนอกจากไปลบไฟล์
    JSON ใน story_saves/ ตรงๆ เอง (เพราะ /story/start ปฏิเสธถ้าเซฟชื่อนี้มีอยู่แล้ว กันการเผลอสร้างซ้ำ)

    ทำไมล้างประวัติบทสนทนาด้วย ไม่ใช่แค่ story_state: ทดสอบจริงพบว่าล้างแค่ story_state เฉยๆ
    โมเดลยังจำบทสนทนาเก่าได้อยู่ (db.get_history อ่านจากตารางแยกต่างหาก) เนื้อเรื่องเลยไม่ได้เริ่ม
    สะอาดจริง ต้องล้างคู่กันเสมอ
    """
    if req.player_character:
        player_character = req.player_character
    elif story_state.exists(session_id):
        player_character = story_state.load(session_id).player_character
    else:
        player_character = {}
    db.reset_session(session_id)
    state = story_state.reset(session_id, player_character)
    return _story_response(state)


@app.get("/story/{session_id}", response_model=StoryStateResponse)
def story_get(session_id: str):
    try:
        state = story_state.load(session_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return _story_response(state)


class StoryAdvanceRequest(BaseModel):
    chapter: int


@app.post("/story/{session_id}/advance", response_model=StoryStateResponse)
def story_advance(session_id: str, req: StoryAdvanceRequest):
    """เลื่อน current_chapter ไปข้างหน้า — เรียกตอนฉากปัจจุบันจบแล้วจริงๆ เท่านั้น

    ทำไมต้องเรียกเอง ไม่ตรวจอัตโนมัติจากบทสนทนา: การเดารอยต่อระหว่างตอนจากข้อความ
    เป็นปัญหา NLP ที่ยากและวัดผลไม่ได้ง่ายๆ ต้นแบบนี้เลยให้ผู้เล่น/GM เป็นคนกดเอง
    ตรงไปตรงมากว่าและ debug ได้ชัดเจนกว่า (ดูเอกสารโปรเจกต์ เฟส 6)
    """
    try:
        state = story_state.load(session_id)
        state = story_state.advance_chapter(state, req.chapter)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _story_response(state)


@app.get("/story/{session_id}/panel", response_class=HTMLResponse)
def story_panel(session_id: str):
    """หน้าเว็บเล็กๆ ดูสถานะเนื้อเรื่อง + กดเลื่อนตอน (เฟส 7, 1 ก.ย. 2569)

    ทำไมต้องมีหน้านี้แทนที่จะใช้ /docs อย่างเดียว: /docs เรียก API ได้แต่ต้องกรอก JSON เองทุกครั้ง
    ไม่เหมาะเวลาต้องเช็ค/กดเลื่อนตอนบ่อยๆ ระหว่างเล่นจริงผ่าน SillyTavern — ที่สำคัญกว่านั้น หน้านี้
    เป็นที่เดียวที่จะเห็นตัวเตือน "ฉากนี้น่าจะจบตอนแล้ว" จาก chapter_watch.py ได้เลย เพราะ SillyTavern
    อ่านได้แค่ field "reply" ตามสคีมา OpenAI เท่านั้น ไม่มีทางเห็น field พิเศษใน ChatResponse

    ไม่ได้ทำเป็น auto-refresh หรืออะไรซับซ้อน — เข้ามารีเฟรชเองเป็นระยะๆ ก็พอ เพราะเป็นโปรเจกต์ส่วนตัว
    """
    try:
        state = story_state.load(session_id)
    except FileNotFoundError:
        return HTMLResponse(
            f"<h1>ไม่พบเซฟ '{session_id}'</h1><p>เรียก POST /story/start ก่อน (ดู /docs)</p>",
            status_code=404,
        )

    hint_html = ""
    if state.chapter_advance_suggested:
        hint_html = f"""
        <div style="background:#fff3cd;border:1px solid #e0b400;padding:12px 16px;border-radius:8px;margin:16px 0;">
          <strong>⚠ ระบบคิดว่าฉากนี้น่าจะจบตอนที่ {state.current_chapter} แล้ว</strong>
          <p style="margin:6px 0 0;color:#555;">{state.chapter_advance_reason or '(ไม่มีเหตุผลระบุ)'}</p>
          <p style="margin:6px 0 0;color:#999;font-size:0.85em;">นี่แค่คำแนะนำจาก LLM รอบเล็ก — ไม่แม่นยำเสมอไป ตัดสินใจเองว่าจะเลื่อนจริงไหม</p>
        </div>
        """

    divergence_rows = "".join(
        f"<li>[ตอนที่ {f.chapter}] {f.fact}</li>" for f in state.divergence_log
    ) or "<li style='color:#999'>ยังไม่มี</li>"

    pc_name = state.player_character.get("name", "(ไม่ได้ตั้งชื่อ)")

    return HTMLResponse(f"""
<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="utf-8">
<title>เลื่อนตอน — {session_id}</title>
<style>
  body {{ font-family: "Segoe UI", sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; color: #222; }}
  button {{ font-size: 1.05em; padding: 10px 22px; cursor: pointer; background: #2c3e50; color: white; border: none; border-radius: 6px; }}
  button:hover {{ background: #1a252f; }}
  input {{ font-size: 1.05em; padding: 6px 8px; width: 70px; }}
  ul {{ padding-left: 20px; }}
  li {{ margin-bottom: 4px; }}
</style>
</head>
<body>
  <h2>สถานะเนื้อเรื่อง — {session_id}</h2>
  <p>ตอนปัจจุบัน: <strong>{state.current_chapter}</strong> (เพดานต้นแบบตอนที่ {story_state.PROTOTYPE_MAX_CHAPTER})</p>
  <p>ตัวละครผู้เล่น: <strong>{pc_name}</strong></p>
  {hint_html}
  <form onsubmit="advance(event)">
    <label>เลื่อนไปตอนที่:
      <input id="chapterInput" type="number" min="{state.current_chapter}" value="{state.current_chapter + 1}">
    </label>
    <button type="submit">เลื่อนตอน</button>
  </form>
  <p id="status"></p>
  <h3>สิ่งที่เบี่ยงไปจากต้นฉบับแล้ว ({len(state.divergence_log)})</h3>
  <ul>{divergence_rows}</ul>
  <script>
    async function advance(e) {{
      e.preventDefault();
      const ch = document.getElementById('chapterInput').value;
      const statusEl = document.getElementById('status');
      statusEl.style.color = '#555';
      statusEl.textContent = 'กำลังเลื่อน...';
      try {{
        const res = await fetch('/story/{session_id}/advance', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{chapter: parseInt(ch, 10)}})
        }});
        if (res.ok) {{
          location.reload();
        }} else {{
          const err = await res.json();
          statusEl.style.color = 'red';
          statusEl.textContent = 'ผิดพลาด: ' + (err.detail || res.status);
        }}
      }} catch (e) {{
        statusEl.style.color = 'red';
        statusEl.textContent = 'ต่อเซิร์ฟเวอร์ไม่ได้: ' + e;
      }}
    }}
  </script>
</body>
</html>
""")


def run_chat(req: ChatRequest) -> ChatResponse:
    """เนื้อหาจริงของ /chat — แยกออกมาจาก route handler เพื่อให้ endpoint อื่นเรียกใช้ได้

    ทำไมต้องแยก (เฟส 7, 31 ส.ค. 2569): app/sillytavern.py ต้องการยิงคำขอผ่านไปป์ไลน์เดียวกับ
    /chat เป๊ะๆ (RAG + guards + divergence ledger ครบ) ไม่ใช่เขียน logic ซ้ำอีกชุด — การแยกฟังก์ชัน
    ธรรมดาแบบนี้พอแล้ว ไม่ต้องเปลี่ยนโครงสร้างอะไรเพิ่ม เพราะ FastAPI route ก็เป็นแค่ฟังก์ชัน python
    ธรรมดาที่รับ req: ChatRequest คืน ChatResponse อยู่แล้ว
    """
    character = load_character(req.character_id)

    # ---- เฟส 6: โหลดเซฟถ้าผูกไว้ ----
    #
    # ไม่ใส่ story_session แปลว่าเรียกแบบเดิมทุกประการ — story ยังเป็น None ตลอดฟังก์ชันนี้
    # ทำให้พฤติกรรมของ tests/benchmark.py ไม่เปลี่ยนแม้แต่น้อย
    story: Optional[story_state.StoryState] = None
    if req.story_session:
        try:
            story = story_state.load(req.story_session)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

    # ตัวละครในเรื่องของผู้เล่น (ถ้ามีเซฟผูกอยู่) ใช้แทนชื่อจริงตอนสร้าง prompt/stop sequence
    # เพราะสิ่งที่ต้องกันไม่ให้ AI เขียนแทน คือ "ตัวละครในเรื่อง" ไม่ใช่ตัวผู้เล่นเอง
    player_display_name = req.player_name
    extra_fields = {}
    if story is not None:
        pc = story.player_character or {}
        if pc.get("name"):
            player_display_name = pc["name"]
        extra_fields = {
            "player_character_name": pc.get("name", req.player_name),
            "player_character_description": pc.get("description", ""),
            "current_chapter": story.current_chapter,
        }

    system_prompt = build_system_prompt(character, player_display_name, full=req.use_prompt,
                                         extra_fields=extra_fields)

    history = db.get_history(req.session_id)
    stop_sequences = guards.get_stop_sequences(player_display_name) if req.use_stop_sequences else None

    # ตรวจขาเข้า: ผู้ใช้พยายามเขียนบทให้ตัวละครหรือเปล่า
    # ตรวจทุกครั้งเหมือนฝั่งขาออก ต่างกันแค่จะแนบหมายเหตุไปให้โมเดลไหม
    character_name = character.get("display_name", "")
    imp = guards.detect_impersonation(req.message, character_name)

    message_to_model = req.message
    if imp.triggered and req.use_input_guard:
        message_to_model = req.message + guards.build_impersonation_note(character_name)

    # ---- ค้นคลังนิยายแล้วแนบต่อท้าย system prompt ----
    #
    # ต่อท้ายหลัง build_system_prompt() ทำงานเสร็จแล้ว ไม่ส่งผ่าน .format()
    # เพราะตัวบทนิยายอาจมีวงเล็บปีกกาปนอยู่ ซึ่งจะทำให้ format() พัง
    # หรือแย่กว่านั้นคือเอาข้อความในนิยายไปแทนตัวแปรใน template มั่วๆ
    rag_used, rag_query, rag_chapters, rag_ms = False, "", [], 0
    block = ""   # เฟส 7: ต้อง init ไว้นอก if เพราะ chapter_watch เอาไปใช้ต่อเป็น chapter_context ด้วย
    if req.use_rag and rag.is_ready():
        rag_query = rag.build_query(req.message, history)
        t_rag = time.perf_counter()
        try:
            max_chapter = story.current_chapter if story is not None else None
            hits = rag.search(rag_query, k=req.rag_k, max_per_chapter=2, max_chapter=max_chapter)
        except Exception as e:
            # ค้นไม่ได้ไม่ควรทำให้ทั้งเทิร์นพัง — ตอบแบบไม่มีบันทึกไปก่อน
            # ตัวละครถูกสอนไว้แล้วว่าถ้าไม่มีบันทึกให้บอกว่าไม่มี ไม่ใช่เดา
            print(f"[chat][warn] ค้นคลังนิยายไม่สำเร็จ: {type(e).__name__}: {e}")
            hits = []
        rag_ms = int((time.perf_counter() - t_rag) * 1000)
        rag_chapters = [h["chapter"] for h in hits]
        block = rag.build_context_block(hits)
        if block:
            system_prompt = system_prompt + "\n" + block
            rag_used = True

    # ---- เฟส 6: แนบ divergence_log ต่อท้ายสุด (มีน้ำหนักเหนือบันทึกที่ค้นมาได้) ----
    if story is not None:
        div_block = story_state.divergence_log_block(story)
        if div_block:
            system_prompt = system_prompt + "\n" + div_block

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

    # ตรวจทุกครั้ง แต่ตัดเฉพาะตอนเปิด guard — ใช้ player_display_name (ตัวละครในเรื่อง)
    # ไม่ใช่ req.player_name เพราะสิ่งที่ต้องกันคือการเขียนแทนตัวละครในเรื่อง
    result = guards.detect_player_narration(raw_reply, player_display_name)
    reply = result.text if req.use_guard else raw_reply

    # บันทึกข้อความต้นฉบับของผู้ใช้ ไม่ใช่ตัวที่แนบหมายเหตุ
    # ไม่งั้นหมายเหตุจะค้างอยู่ในประวัติแล้วถูกส่งซ้ำทุกเทิร์นถัดไป
    db.log_turn(req.session_id, "player", req.message)
    db.log_turn(req.session_id, "character", reply)

    # ---- เฟส 6: สกัด divergence fact แล้วบันทึกเข้าเซฟ ----
    #
    # ทำหลังบันทึกประวัติบทสนทนาแล้ว เพื่อให้ turn_index ที่เก็บตรงกับความยาวประวัติจริง
    # ล้มเหลวได้โดยไม่ทำให้ทั้งเทิร์นพัง (ดูเหตุผลใน divergence.py)
    divergence_detected, divergence_fact = False, ""
    if story is not None and req.use_divergence_extraction:
        check = divergence.check_turn(req.message, reply, story.current_chapter)
        if check.diverges:
            divergence_detected = True
            divergence_fact = check.fact
            story_state.add_divergence_fact(
                story, fact=check.fact,
                turn_index=len(db.get_history(req.session_id)),
                chapter=story.current_chapter,
            )

    # ---- เฟส 7: ประเมินว่าฉากนี้น่าจะจบตอนหรือยัง (แค่เตือน ไม่เลื่อนเอง — ดู chapter_watch.py) ----
    #
    # เก็บผลลงเซฟเสมอ (แม้ผลจะเป็น False) ไม่ใช่แค่ตอน likely_concluded=True เพราะถ้าฉากขยับต่อไป
    # แล้วไม่จบแล้วจริงๆ ต้องล้างตัวเตือนเก่าที่อาจค้างอยู่ในหน้า /panel ด้วย ไม่งั้นจะเห็นตัวเตือน
    # ของเทิร์นก่อนหน้าค้างอยู่ทั้งที่สถานการณ์เปลี่ยนไปแล้ว
    chapter_advance_suggested, chapter_advance_reason = False, ""
    if story is not None and req.use_chapter_watch:
        progress = chapter_watch.check_progress(reply, story.current_chapter, chapter_context=block)
        chapter_advance_suggested = progress.likely_concluded
        chapter_advance_reason = progress.reason
        story_state.set_chapter_advance_hint(story, chapter_advance_suggested, chapter_advance_reason)

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
        rag_used=rag_used,
        rag_query=rag_query,
        rag_chapters=rag_chapters,
        rag_ms=rag_ms,
        story_chapter=story.current_chapter if story is not None else None,
        divergence_detected=divergence_detected,
        divergence_fact=divergence_fact,
        chapter_advance_suggested=chapter_advance_suggested,
        chapter_advance_reason=chapter_advance_reason,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return run_chat(req)
