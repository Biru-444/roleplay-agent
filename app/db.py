"""
db.py — บันทึกและดึงประวัติบทสนทนา

ถ้าตั้ง DATABASE_URL ไว้    -> เก็บลง PostgreSQL (ใช้ Supabase / Neon free tier ได้ ไม่ต้องตั้ง server เอง)
ถ้าไม่ได้ตั้ง                -> เก็บไว้ในหน่วยความจำของโปรเซสแทน

ที่ต้องมีโหมดหน่วยความจำ เพราะอยากให้รัน tests/benchmark.py เก็บตัวเลขได้ทันที
โดยไม่ต้องรอตั้งฐานข้อมูลให้เสร็จก่อน — ข้อมูลจะหายเมื่อปิดเซิร์ฟเวอร์ ซึ่งสำหรับ
การวัดผลรอบเดียวจบถือว่ายอมรับได้

ตอนนี้ทำแค่ "บันทึกและดึงประวัติ" เท่านั้น ยังไม่ใช่ระบบความจำ 3 ชั้นเต็มรูปแบบ
(ชั้นสถานะปัจจุบันแบบ JSON และชั้นความจำระยะยาวที่ค้นคืนตามบริบท อยู่ใน Roadmap ของ README)
"""

import os
from collections import defaultdict
from typing import Dict, List

DATABASE_URL = os.environ.get("DATABASE_URL")
USING_POSTGRES = bool(DATABASE_URL)

_memory_store: Dict[str, List[Dict[str, str]]] = defaultdict(list)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS conversation_turns (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,          -- 'player' หรือ 'character'
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_conversation_turns_session
    ON conversation_turns (session_id, id);
"""


def _get_connection():
    import psycopg2  # import ตรงนี้ เพื่อให้โหมดหน่วยความจำไม่ต้องติดตั้ง psycopg2

    return psycopg2.connect(DATABASE_URL)


def init_db() -> str:
    """คืนข้อความบอกว่ากำลังใช้โหมดไหน เอาไปพิมพ์ตอน startup"""
    if not USING_POSTGRES:
        return "ไม่พบ DATABASE_URL — ใช้โหมดเก็บประวัติในหน่วยความจำ (ข้อมูลหายเมื่อปิดเซิร์ฟเวอร์)"

    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
        conn.commit()
    return "เชื่อมต่อ PostgreSQL แล้ว"


def log_turn(session_id: str, role: str, content: str) -> None:
    if not USING_POSTGRES:
        _memory_store[session_id].append({"role": role, "content": content})
        return

    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversation_turns (session_id, role, content) VALUES (%s, %s, %s)",
                (session_id, role, content),
            )
        conn.commit()


def get_history(session_id: str, limit: int = 15) -> List[Dict[str, str]]:
    if not USING_POSTGRES:
        return list(_memory_store[session_id][-limit:])

    from psycopg2.extras import RealDictCursor

    with _get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT role, content FROM conversation_turns
                WHERE session_id = %s
                ORDER BY id DESC
                LIMIT %s
                """,
                (session_id, limit),
            )
            rows = cur.fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def reset_session(session_id: str) -> None:
    """ล้างประวัติของ session หนึ่ง — benchmark ใช้เริ่มการทดสอบแต่ละเงื่อนไขจากศูนย์"""
    if not USING_POSTGRES:
        _memory_store.pop(session_id, None)
        return

    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM conversation_turns WHERE session_id = %s", (session_id,))
        conn.commit()
