"""
chapter_watch.py — ตรวจว่าฉากปัจจุบัน "น่าจะ" จบตอนแล้วหรือยัง (เฟส 7, 1 ก.ย. 2569)

ทำไมทำเป็น "ตัวเตือน" ไม่ใช่เลื่อนตอนอัตโนมัติ (ตกลงกับผู้ใช้ไว้ตอนคุยออกแบบเรื่องนี้):
  ความเสี่ยงของฟีเจอร์นี้กับ divergence.py ไม่เท่ากัน — ถ้า divergence_fact สกัดผิด (เช่น Bug 5 ที่
  เจอในเฟส 6) แค่ข้อความสรุปในเซฟผิด แก้/ลบทีหลังได้ ไม่มีอะไรเสียหายถาวร
  แต่ถ้า "เลื่อนตอนผิด" (ตัดสินว่าจบตอนทั้งที่จริงยังไม่จบ) RAG จะเปิดให้ค้นเนื้อหาตอนถัดไปทันที
  (ดู main.py: max_chapter = story.current_chapter) และผู้บรรยายอาจหยิบมาเล่าให้ผู้เล่นอ่านไปแล้ว
  ในเทิร์นนั้นเลย — สปอยล์ตัวเองที่ย้อนกลับไม่ได้ ต่อให้ตั้ง current_chapter กลับที่เดิมทีหลัง ผู้เล่น
  ก็อ่านไปแล้ว เพราะงั้นเลือกทำแค่ "เตือน" ให้ผู้เล่น/GM เป็นคนตัดสินใจเองขั้นสุดท้ายเสมอ ไม่ใช่ให้
  ระบบตัดสินใจแทน (ดูหน้า /story/{id}/panel ใน main.py ที่แสดงตัวเตือนนี้)

สถาปัตยกรรมเหมือน divergence.py ทุกอย่าง (call เล็กแยกต่างหากหลังตอบเทิร์นหลัก, response_schema
บังคับรูปแบบผลลัพธ์ที่ระดับ API แทนการขอด้วยข้อความ, ล้มเหลวได้โดยไม่ทำให้ทั้งเทิร์นพัง) ต่างกันแค่
คำถามที่ถาม — เหตุผลของแต่ละจุดแบบเต็มๆ ดูได้ในไฟล์นั้น ไม่ต้องอธิบายซ้ำที่นี่
"""
from dataclasses import dataclass

from pydantic import BaseModel

from app import llm


class _ChapterProgressResult(BaseModel):
    likely_concluded: bool
    reason: str   # เหตุผลสั้นๆ ว่าทำไมคิดว่าจบ/ยังไม่จบตอน — ให้ผู้เล่นอ่านประกอบการตัดสินใจเอง


_SYSTEM_PROMPT = (
    "คุณคือระบบตรวจสอบเบื้องหลัง ไม่ใช่ตัวละครหรือผู้บรรยายในเรื่อง\n"
    "งานของคุณ: อ่านคำตอบล่าสุดของผู้บรรยายหนึ่งเทิร์น เทียบกับบันทึกต้นฉบับของตอนที่ระบุ (ถ้ามีแนบมา)\n"
    "แล้วประเมินว่า 'เหตุการณ์หลักของตอนนี้ตามต้นฉบับ' น่าจะดำเนินจบแล้วหรือยัง (likely_concluded)\n"
    "\n"
    "นี่เป็นแค่ 'คำแนะนำ' ให้ผู้เล่นไปตัดสินใจเองว่าจะเลื่อนตอนจริงไหม ไม่ใช่การตัดสินขั้นสุดท้าย\n"
    "เพราะงั้นถ้าไม่มั่นใจ ให้ตอบ likely_concluded=false ไว้ก่อนเสมอ — เตือนก่อนเวลาแค่เสียเวลาผู้เล่น\n"
    "ไปตรวจดูเฉยๆ แต่เตือนช้าเกินไปหรือไม่เตือนเลยอาจทำให้พลาดจังหวะเลื่อนตอน ไม่ใช่ความเสี่ยงที่ต่างกัน\n"
    "มาก จึงเลือกความระมัดระวังไว้ก่อนเสมอเวลาไม่แน่ใจ\n"
    "\n"
    "ตอบ likely_concluded=true ก็ต่อเมื่อคำตอบล่าสุดเล่าถึงเหตุการณ์ที่เป็นจุดจบฉาก/จุดเปลี่ยนฉากใหญ่ของ\n"
    "ตอนนี้อย่างชัดเจนตามบันทึกต้นฉบับที่แนบมา (เช่นฉากจบลงจริง เปลี่ยนสถานที่ใหญ่ ข้ามเวลาไปมาก\n"
    "เหตุการณ์หลักของตอนคลี่คลายแล้ว) ไม่ใช่แค่บทสนทนาดำเนินไปตามปกติในฉากเดิม\n"
    "เขียน reason สั้นๆ ไม่เกิน 20 คำ อธิบายเหตุผลที่ตัดสินแบบนั้น ให้ผู้เล่นอ่านแล้วตัดสินใจเองต่อได้"
)


@dataclass
class ChapterProgressCheck:
    likely_concluded: bool
    reason: str = ""


def check_progress(reply: str, chapter: int, chapter_context: str = "") -> ChapterProgressCheck:
    """เรียก LLM รอบเล็กประเมินว่าฉากตอนที่ {chapter} น่าจะจบหรือยัง

    ล้มเหลวได้โดยไม่ทำให้ทั้งเทิร์นพัง (แนวทางเดียวกับ divergence.check_turn) — คืน
    likely_concluded=False เงียบๆ ถ้า LLM ล่ม ตอบว่าง หรือ parse ไม่ได้ เพราะฟีเจอร์นี้เป็นแค่
    ตัวช่วยเตือน ไม่ใช่กลไกหลักของเกม พลาดไปหนึ่งเทิร์นไม่กระทบอะไรมาก
    """
    user_message = (
        f"ตอนที่ {chapter} ของเนื้อเรื่องต้นฉบับ\n"
        f"บันทึกต้นฉบับของตอนนี้ที่มี (อาจไม่ครบทั้งตอน): {chapter_context or '(ไม่มีบันทึกแนบมา)'}\n"
        f"คำตอบล่าสุดของผู้บรรยาย: {reply}"
    )
    try:
        result = llm.generate_response(
            system_prompt=_SYSTEM_PROMPT,
            history=[],
            user_message=user_message,
            temperature=0.2,       # งานนี้ต้องการความคงเส้นคงวา ไม่ใช่ความสร้างสรรค์แบบบทสนทนาหลัก
            max_output_tokens=200,
            response_mime_type="application/json",
            response_schema=_ChapterProgressResult,
        )
    except Exception as e:
        print(f"[chapter_watch][warn] ตรวจไม่สำเร็จ: {type(e).__name__}: {e}")
        return ChapterProgressCheck(likely_concluded=False)

    text = (result.text or "").strip()
    if not text:
        return ChapterProgressCheck(likely_concluded=False)

    try:
        parsed = _ChapterProgressResult.model_validate_json(text)
    except Exception as e:
        print(f"[chapter_watch][warn] parse JSON ไม่สำเร็จ: {e} — raw={text!r}")
        return ChapterProgressCheck(likely_concluded=False)

    return ChapterProgressCheck(likely_concluded=parsed.likely_concluded, reason=parsed.reason.strip())
