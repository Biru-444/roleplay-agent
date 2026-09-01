"""
story_state.py — ไฟล์ save ต่อเซฟ สำหรับเฟส 6 (ระบบเนื้อเรื่องแยกสาขา / divergence ledger)

หน้าที่: เก็บ "ผู้เล่นอยู่ตอนไหนแล้ว" (current_chapter), ข้อมูลตัวละครที่ผู้เล่นสร้างเอง
(player_character) และ divergence_log (รายการเหตุการณ์ที่ถูกเปลี่ยนไปจากต้นฉบับ)

ทำไมเก็บเป็นไฟล์ JSON แยกจาก db.py (ที่เก็บประวัติบทสนทนา):
  - ประวัติบทสนทนาคือ "บทพูดดิบ" ยาวเป็นร้อยเทิร์น ไม่เหมาะจะยัดเข้า prompt ทุกครั้ง
  - divergence_log คือ "สรุปข้อเท็จจริงสั้นๆ ที่เปลี่ยนไป" ไม่กี่สิบบรรทัดตลอดทั้งเรื่อง
    (ต้นแบบนี้จำกัดแค่ ~20-30 ตอนแรก) ยัดเข้า prompt ตรงๆ ได้เลยโดยไม่ต้องมี
    retrieval index ที่สอง — ตามสถาปัตยกรรมที่ตกลงกันไว้ในเอกสารโปรเจกต์ เฟส 6
  - แยกไฟล์ยังทำให้ debug ง่าย เปิด JSON ดูตรงๆ ได้ว่าเรื่องเบี่ยงจากต้นฉบับตรงไหนบ้าง
    โดยไม่ต้องไล่อ่านบทสนทนาทั้งหมด

เก็บเป็นไฟล์ ไม่ใช่ตาราง Postgres เพราะขนาดข้อมูลต่อเซฟเล็กมาก (ไม่กี่ KB)
และนี่คือต้นแบบพิสูจน์แนวคิด ยังไม่ต้องรองรับผู้เล่นพร้อมกันจำนวนมากที่ query ร่วมกัน
ถ้าจะขยายเป็นระบบจริงทีหลัง ย้ายมาที่ db.py (ตาราง story_states) ได้โดยไม่ต้องแตะ
โครงสร้าง StoryState เลย — เปลี่ยนแค่ load()/save() สองฟังก์ชันนี้
"""
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# โฟลเดอร์เก็บไฟล์ save — ตั้งทับได้ด้วย env var เหมือนไฟล์อื่นๆ ในโปรเจกต์ (ดู rag.py)
_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "story_saves"
SAVES_DIR = Path(os.environ.get("STORY_SAVES_DIR", str(_DEFAULT_DIR)))

# เพดานตอนสำหรับต้นแบบนี้ — ตกลงกันไว้ว่าพิสูจน์แนวคิดแค่ 20-30 ตอนแรกก่อน
# ตั้งเป็นค่าคงที่จุดเดียว ถ้าจะขยายขอบเขตทีหลังแก้เลขนี้เลขเดียวจบ (ดูเอกสารโปรเจกต์ เฟส 6)
PROTOTYPE_MAX_CHAPTER = int(os.environ.get("STORY_PROTOTYPE_MAX_CHAPTER", "30"))


@dataclass
class DivergenceFact:
    chapter: int          # ตอนที่ (อ้างอิงตามเนื้อเรื่องต้นฉบับ) ที่เหตุการณ์นี้เกิด
    fact: str              # ข้อเท็จจริงสั้นๆ เช่น "ผู้เล่นเตือนไคลน์ล่วงหน้าเรื่องจดหมายลึกลับ"
    turn_index: int        # เทิร์นที่เท่าไหร่ของบทสนทนา (ไว้ตรวจสอบย้อนหลังว่าเกิดตอนไหน)
    created_at: float = field(default_factory=time.time)


@dataclass
class StoryState:
    session_id: str
    current_chapter: int = 1
    player_character: Dict[str, str] = field(default_factory=dict)  # เช่น {"name":..., "description":...}
    divergence_log: List[DivergenceFact] = field(default_factory=list)

    # ---- เฟส 7: ผลตรวจล่าสุดจาก chapter_watch.py ----
    #
    # ทำไมเก็บลงเซฟแทนที่จะส่งกลับใน ChatResponse อย่างเดียว: SillyTavern อ่านได้แค่ field
    # "reply" ตามสคีมา OpenAI เท่านั้น (ดู app/sillytavern.py) ไม่มีทางเห็น field พิเศษอื่นๆ ใน
    # ChatResponse เลย ต้องมีที่เก็บถาวรแยกต่างหาก ให้หน้ากดเลื่อนตอน (/story/{id}/panel ใน main.py)
    # อ่านมาแสดงได้ แม้ผู้เล่นจะคุยผ่าน SillyTavern ทั้งหมดโดยไม่เห็น field พวกนี้เลยก็ตาม
    chapter_advance_suggested: bool = False
    chapter_advance_reason: str = ""

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "StoryState":
        facts = [DivergenceFact(**f) for f in d.get("divergence_log", [])]
        return cls(
            session_id=d["session_id"],
            current_chapter=d.get("current_chapter", 1),
            player_character=d.get("player_character", {}),
            divergence_log=facts,
            # .get() กับค่าเริ่มต้น เพราะเซฟเก่าก่อนเฟส 7 จะไม่มีสอง field นี้เลย — ไม่ให้ load() พัง
            chapter_advance_suggested=d.get("chapter_advance_suggested", False),
            chapter_advance_reason=d.get("chapter_advance_reason", ""),
        )


def _path(session_id: str) -> Path:
    # กัน path traversal เบื้องต้น เพราะ session_id มาจาก request body ของภายนอก
    safe_id = "".join(c for c in session_id if c.isalnum() or c in ("-", "_"))
    if not safe_id:
        raise ValueError("session_id ต้องมีตัวอักษร/ตัวเลข/เครื่องหมาย -_ อย่างน้อยหนึ่งตัว")
    return SAVES_DIR / f"{safe_id}.json"


def exists(session_id: str) -> bool:
    return _path(session_id).exists()


def load(session_id: str) -> StoryState:
    p = _path(session_id)
    if not p.exists():
        raise FileNotFoundError(f"ไม่พบเซฟ '{session_id}' — เรียก /story/start ก่อน")
    return StoryState.from_json(json.loads(p.read_text(encoding="utf-8")))


def load_or_create(session_id: str, player_character: Optional[Dict[str, str]] = None) -> StoryState:
    if exists(session_id):
        return load(session_id)
    state = StoryState(session_id=session_id, player_character=player_character or {})
    save(state)
    return state


def reset(session_id: str, player_character: Optional[Dict[str, str]] = None) -> StoryState:
    """ล้างเซฟกลับไปเป็นค่าเริ่มต้น (ตอนที่ 1, divergence_log ว่าง) โดยไม่ต้องลบไฟล์เอง (เฟส 7)

    ทำไมต้องมีฟังก์ชันนี้แยกจาก load_or_create: เดิมถ้าอยากเริ่มทดสอบสถานการณ์ใหม่แบบสะอาด
    ต้องลบไฟล์ JSON ในโฟลเดอร์ story_saves/ ตรงๆ เพราะ /story/start ปฏิเสธถ้าเซฟชื่อนี้มีอยู่แล้ว
    (กันการเผลอสร้างซ้ำ) เมธอดนี้แทนที่เซฟทั้งก้อนด้วยของใหม่ตรงๆ ไม่ผ่านเช็ค exists() แบบ start()
    เรียกผ่าน POST /story/{session_id}/reset ใน main.py (ซึ่งล้างประวัติบทสนทนาใน db.py คู่กันด้วย
    เพราะล้างแค่ story_state เฉยๆ โมเดลจะยังจำบทสนทนาเก่าได้อยู่ ไม่เริ่มสะอาดจริง)
    """
    state = StoryState(session_id=session_id, player_character=player_character or {})
    save(state)
    return state


def save(state: StoryState) -> None:
    SAVES_DIR.mkdir(parents=True, exist_ok=True)
    p = _path(state.session_id)
    p.write_text(json.dumps(state.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")


def advance_chapter(state: StoryState, chapter: int) -> StoryState:
    """เลื่อนตอนปัจจุบันไปข้างหน้าเท่านั้น

    ตามที่ตกลงกันไว้ (ตำแหน่งในเรื่อง): ผู้เล่นเริ่มตอน 1 แล้วไล่ไปเรื่อยๆ ตามลำดับ ไม่มีการกระโดดข้าม
    เมธอดนี้เลยปฏิเสธการตั้งค่าย้อนหลัง และปฏิเสธการตั้งเกินเพดานต้นแบบ (PROTOTYPE_MAX_CHAPTER)
    """
    if chapter < state.current_chapter:
        raise ValueError(
            f"ถอยหลังไม่ได้ — ตอนปัจจุบันคือ {state.current_chapter} จะตั้งเป็น {chapter} ไม่ได้"
        )
    if chapter > PROTOTYPE_MAX_CHAPTER:
        raise ValueError(f"ต้นแบบนี้รองรับถึงตอนที่ {PROTOTYPE_MAX_CHAPTER} เท่านั้น")
    state.current_chapter = chapter
    # เลื่อนตอนแล้ว ตัวเตือนเดิม (ถ้ามี) ล้าสมัยทันที ไม่ล้างไว้จะเห็นตัวเตือนของตอนเก่าค้างอยู่
    # ในหน้า /story/{id}/panel ทั้งที่เพิ่งเลื่อนไปแล้ว
    state.chapter_advance_suggested = False
    state.chapter_advance_reason = ""
    save(state)
    return state


def set_chapter_advance_hint(state: StoryState, suggested: bool, reason: str) -> StoryState:
    """บันทึกผลตรวจล่าสุดจาก chapter_watch.py ลงเซฟ — ดูเหตุผลเรื่องทำไมต้องเก็บถาวรใน docstring

    ของ field chapter_advance_suggested/reason ด้านบน (สรุปสั้นๆ: SillyTavern ไม่เห็น field
    พิเศษใน ChatResponse เลย ต้องมีที่เก็บถาวรแยกให้หน้ากดเลื่อนตอนอ่านได้)
    """
    state.chapter_advance_suggested = suggested
    state.chapter_advance_reason = reason
    save(state)
    return state


def add_divergence_fact(state: StoryState, fact: str, turn_index: int,
                         chapter: Optional[int] = None) -> StoryState:
    state.divergence_log.append(
        DivergenceFact(chapter=chapter if chapter is not None else state.current_chapter,
                        fact=fact, turn_index=turn_index)
    )
    save(state)
    return state


def divergence_log_block(state: StoryState) -> str:
    """แปลง divergence_log เป็นบล็อกข้อความต่อท้าย system prompt

    ยัดทั้ง log ใส่ตรงๆ ไม่ต้องมี retrieval index ที่สอง เพราะต้นแบบนี้จำกัดแค่
    20-30 ตอนแรก log จึงยาวไม่กี่สิบบรรทัดตลอดทั้งเรื่อง (เหตุผลเต็มอยู่ในเอกสารโปรเจกต์ เฟส 6
    หัวข้อทำไมไม่เลือก option A — full simulation)
    """
    if not state.divergence_log:
        return ""
    lines = [
        "",
        "## สิ่งที่เบี่ยงไปจากเนื้อเรื่องต้นฉบับแล้ว (สำคัญที่สุด — มีน้ำหนักเหนือบันทึกที่ค้นมาได้)",
        "",
        "เหตุการณ์ต่อไปนี้ถูกผู้เล่นเปลี่ยนไปจากต้นฉบับ ถือว่าเกิดขึ้นจริงในเนื้อเรื่องนี้แล้ว",
        "ถ้าขัดแย้งกับส่วน \"บันทึกที่ค้นมาได้\" (ซึ่งเป็นเนื้อหาต้นฉบับ) ให้ยึดรายการนี้เป็นหลัก",
        "ห้ามเล่าเหตุการณ์ต้นฉบับที่ถูกเปลี่ยนไปแล้วราวกับว่ายังเป็นแบบเดิม",
        "",
    ]
    for f in state.divergence_log:
        lines.append(f"- [ตอนที่ {f.chapter}] {f.fact}")
    return "\n".join(lines)
