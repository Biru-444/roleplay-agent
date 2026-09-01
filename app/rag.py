"""
rag.py — สะพานระหว่าง FastAPI กับระบบค้นหาคลังนิยาย (เฟส 5 ขั้นที่ 1)

หน้าที่ของไฟล์นี้มีอย่างเดียว: ทำให้ `Retriever` ใน noveldata/phase3_scripts/retrieve.py
ถูกโหลด "ครั้งเดียวตอนสตาร์ทเซิร์ฟเวอร์" แล้วใช้ซ้ำได้ทุก request

ทำไมต้องมีไฟล์นี้แทนที่จะ import ตรงๆ:
  1. retrieve.py กับไฟล์ index อยู่คนละโฟลเดอร์กับ app/ ต้องต่อ path ให้
  2. ถ้า import ตรงๆ ใน main.py แล้วไฟล์ index ยังไม่มี เซิร์ฟเวอร์จะสตาร์ทไม่ขึ้นทั้งตัว
     ทั้งที่ endpoint อื่น (/chat, /health) ไม่ได้ใช้ RAG เลย — ไฟล์นี้เลยจับ error ไว้
     แล้วให้เซิร์ฟเวอร์รันต่อได้ในโหมด "ไม่มี RAG" (เหมือนที่ db.py ทำกับ Postgres)

ตั้งค่าที่อยู่ index ได้ด้วย env var:
    NOVEL_INDEX_DIR=D:\project Resume\noveldata\phase3_scripts
ถ้าไม่ตั้ง จะเดาจาก path ของโปรเจกต์เอง
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# โฟลเดอร์ที่มี retrieve.py + ไฟล์ index — ค่าเริ่มต้นคือ ../noveldata/phase3_scripts
_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "noveldata" / "phase3_scripts"
INDEX_DIR = Path(os.environ.get("NOVEL_INDEX_DIR", str(_DEFAULT_DIR)))

_retriever = None          # ตัวจริง ถ้าโหลดสำเร็จ
_load_error: str = ""      # ข้อความ error ถ้าโหลดไม่ได้ (เอาไปโชว์ใน /health)
_load_seconds: float = 0.0


def is_ready() -> bool:
    return _retriever is not None


def status() -> dict:
    """สถานะสำหรับ /health — บอกให้รู้ว่า RAG พร้อมไหม ถ้าไม่พร้อมเพราะอะไร"""
    return {
        "ready": is_ready(),
        "index_dir": str(INDEX_DIR),
        "chunks": len(_retriever.meta) if _retriever else 0,
        "load_seconds": round(_load_seconds, 1),
        "error": _load_error,
    }


def warmup() -> str:
    """โหลด index + โมเดล — เรียกครั้งเดียวตอน startup

    คืนข้อความสรุปเพื่อ print ลง log ไม่ raise ออกไปข้างนอก
    เพราะไม่อยากให้เซิร์ฟเวอร์ทั้งตัวตายเพราะ RAG ไม่พร้อม
    """
    global _retriever, _load_error, _load_seconds

    if _retriever is not None:
        return "RAG: โหลดไว้แล้ว"

    if not INDEX_DIR.exists():
        _load_error = f"ไม่เจอโฟลเดอร์ {INDEX_DIR}"
        return f"RAG: ปิดใช้งาน — {_load_error}"

    t0 = time.time()
    try:
        # เพิ่ม path ให้ import retrieve.py ได้ (มันอยู่คนละโฟลเดอร์กับ app/)
        if str(INDEX_DIR) not in sys.path:
            sys.path.insert(0, str(INDEX_DIR))
        from retrieve import Retriever

        _retriever = Retriever(data_dir=str(INDEX_DIR), verbose=True)
        _load_seconds = time.time() - t0
        _load_error = ""
        return (f"RAG: พร้อมใช้งาน — {len(_retriever.meta):,} ชิ้น "
                f"(ใช้เวลาโหลด {_load_seconds:.0f} วินาที)")
    except Exception as e:
        _load_seconds = time.time() - t0
        _load_error = f"{type(e).__name__}: {e}"
        _retriever = None
        return f"RAG: ปิดใช้งาน — {_load_error}"


def search(query: str, k: int = 5, max_per_chapter: Optional[int] = 2,
           max_chapter: Optional[int] = None) -> List[dict]:
    """max_chapter: กันสปอยล์ (เฟส 6) — ส่งต่อ current_chapter ของเซฟเพื่อไม่ให้ค้นเจอตอนที่ยังไม่ถึง"""
    if _retriever is None:
        raise RuntimeError(f"RAG ยังไม่พร้อม: {_load_error or 'ยังไม่ได้เรียก warmup()'}")
    return _retriever.search(query, k=k, max_per_chapter=max_per_chapter, max_chapter=max_chapter)


def as_context(query: str, k: int = 5, max_per_chapter: int = 2,
               max_chars: int = 6000, max_chapter: Optional[int] = None) -> str:
    if _retriever is None:
        raise RuntimeError(f"RAG ยังไม่พร้อม: {_load_error or 'ยังไม่ได้เรียก warmup()'}")
    return _retriever.as_context(query, k=k, max_per_chapter=max_per_chapter,
                                 max_chars=max_chars, max_chapter=max_chapter)


# ---------------------------------------------------------------------------
# ประกอบคำค้นจากบทสนทนา
# ---------------------------------------------------------------------------

# ข้อความสั้นกว่านี้ถือว่า "สั้นเกินจะค้นเดี่ยวๆ"
# ที่มาของเลข: ทดสอบจริงแล้วพบว่าคำถามยาวๆ ที่มีรายละเอียดเยอะค้นเจอ
# แต่คำถามสั้นแบบ "ลูเมี่ยนปีนหน้าต่างตอนไหน" ไม่เจอ เพราะ embedding
# ได้สัญญาณน้อยเกินไป และ BM25 ก็ไม่มีคำหายากให้จับ
SHORT_QUERY_CHARS = 30

# เอาข้อความก่อนหน้าของผู้เล่นมาต่อได้กี่ข้อความ
CONTEXT_TURNS = 2


def build_query(message: str, history=None) -> str:
    """ประกอบคำค้นจากข้อความล่าสุด + ข้อความก่อนหน้าถ้าสั้นเกินไป

    ทำไมต้องมี: ผู้เล่นพิมพ์สั้นกว่าคำถามในชุดทดสอบมาก และมักใช้คำแทน
    ("แล้วเขาทำอะไรต่อ") ซึ่งค้นเดี่ยวๆ ไม่ได้ความ การเอาเทิร์นก่อนหน้ามาต่อ
    ช่วยเติมชื่อเฉพาะกลับเข้าไปในคำค้นโดยไม่ต้องเสีย API call เพิ่ม

    ข้อจำกัดที่ต้องรู้: นี่เป็นแค่ heuristic ไม่ได้ฉลาด ถ้าผู้เล่นเปลี่ยนเรื่อง
    กะทันหันด้วยข้อความสั้น มันจะลากเรื่องเก่ามาปนด้วย
    ทางแก้ที่ดีกว่าคือให้ LLM เขียนคำค้นใหม่ให้ แต่นั่นเสียเวลาเพิ่มอีกหนึ่ง
    รอบเรียก API ต่อหนึ่งเทิร์น — ยังไม่ทำจนกว่าจะพิสูจน์ว่า heuristic นี้ไม่พอ
    """
    message = (message or "").strip()
    if len(message) >= SHORT_QUERY_CHARS or not history:
        return message

    prev = [h.get("content", "") for h in history if h.get("role") == "player"]
    if not prev:
        return message
    return " ".join(prev[-CONTEXT_TURNS:] + [message]).strip()


def build_context_block(hits: List[dict]) -> str:
    """แปลงผลค้นเป็นบล็อกข้อความต่อท้าย system prompt

    หมายเหตุสำคัญ: บล็อกนี้ต้อง "ต่อท้าย" prompt ที่ format เสร็จแล้วเท่านั้น
    ห้ามส่งเข้า str.format() เพราะตัวบทนิยายอาจมีวงเล็บปีกกาปนอยู่
    ซึ่งจะทำให้ format() พังหรือเอาข้อความในนิยายไปแทนที่ตัวแปรมั่วๆ
    """
    if not hits:
        return ""

    parts = [
        "",
        "## บันทึกที่ค้นมาได้",
        "",
        "ข้อความข้างล่างนี้คัดมาจากคลังบันทึกจริง ตามคำถามล่าสุดของผู้เล่น",
        "ตอบเรื่องเนื้อหาได้จากตรงนี้เท่านั้น ถ้าไม่พอให้บอกว่าไม่มีบันทึก",
        "",
    ]
    for h in hits:
        parts.append(f"[ตอนที่ {h['chapter']} — {h['chapter_title']}]")
        parts.append(h["text"])
        parts.append("")
    return "\n".join(parts)
