"""
test_phase6.py — ทดสอบระบบเนื้อเรื่องแยกสาขา (divergence ledger) แบบยิงจริงผ่าน HTTP

วิธีใช้ (เหมือน tests/benchmark.py):
    1. เปิดเซิร์ฟเวอร์ไว้ก่อน:  uvicorn app.main:app --reload
    2. อีกหน้าต่างหนึ่งรัน:      python tests/test_phase6.py

ทำไมใช้สคริปต์ python แทน curl: README บอกไว้แล้วว่า curl -d ที่มีภาษาไทยพังบน
Windows terminal (แปลง encoding ผิด) ส่วน python เขียนไฟล์ JSON ตรงๆ ด้วย utf-8
ไม่ผ่าน terminal เลยไม่มีปัญหานี้

ผลทุกขั้นตอนจะถูกเก็บไว้ที่ tests/phase6_results.json — เปิดไฟล์นี้ (ไม่ใช่อ่านจาก
stdout) เพราะ terminal ของ Windows แสดงภาษาไทยเพี้ยน (บทเรียนเดียวกับ benchmark.py)

สคริปต์นี้ตรวจสิ่งที่ตรวจได้อัตโนมัติ (chapter gate, ฟิลด์ story_chapter,
error ตอน advance ผิดกฎ) ส่วน "คำตอบยังคุมโทนเรื่องดีไหม" กับ "จำ divergence
ได้จริงไหมตอนคุยต่อ" ต้องอ่านเองจากไฟล์ผลลัพธ์ — งานนี้ยังไม่มีเฉลยอัตโนมัติแบบ
recall@5 ของมาร์ธา (ดูหัวข้อ "ยังไม่ได้ทำ" ในเอกสารโปรเจกต์ เฟส 6)

ขั้น chat_test_npc_impersonation (ท้ายสคริปต์) ก็ต้องอ่านเองเช่นกัน — ทดสอบว่ากฎข้อ
7-8 ของ narrator_lotm.json (แก้ทิศทางไปแล้วหลังบั๊กที่ 2 ในเอกสารโปรเจกต์) ยังกัน
"ผู้เล่นสั่งแทนตัวละครอื่น" (เช่นไคลน์) ได้อยู่ไหม หลังจากที่แก้ให้ยอมรับสิ่งที่ผู้เล่น
ทำผ่านตัวละครตัวเองแล้ว — ไม่มีทางเช็คเป็น boolean อัตโนมัติได้ (ไม่มี guard ฝั่งขาเข้า
ที่ตรวจเรื่องนี้ เพราะ detect_impersonation() เช็คแค่ชื่อของตัวละครที่กำลังคุยด้วยเอง
คือ "ผู้บรรยาย" ไม่ใช่ "ไคลน์" — ดูโค้ดใน main.py) ต้องอ่านคำตอบเทียบกับข้อความที่ส่งไปเอง

**สำคัญ — ล้างเซฟก่อนรันทุกครั้ง:** พบจริงตอนรันรอบที่ 2 (31 ส.ค. 2569) ว่า `/story/start`
เป็น idempotent (ถ้าเซฟมีอยู่แล้วจะใช้ของเดิมต่อ ไม่สร้างใหม่) ทำให้รันสคริปต์นี้ซ้ำ
โดยไม่ล้างเซฟก่อน จะพา `divergence_log` เก่าจากรอบก่อนหน้าติดมาด้วย — ถ้ารอบก่อนมี
fact ที่ parse ผิด (เช่นตอนเจอบั๊ก response_schema) ค้างอยู่ในนั้น มันจะปนเข้าไปใน
prompt ของรอบใหม่ทุกเทิร์น ทำให้ผลดูสับสนโดยไม่เกี่ยวกับโค้ดที่เพิ่งแก้เลย
สคริปต์นี้จึงลบไฟล์เซฟทิ้งก่อนเริ่มเสมอ เพื่อให้ทุกรอบเปรียบเทียบกันได้ตรงๆ
"""
import json
import sys
from pathlib import Path

import requests

API_URL = "http://localhost:8000"
SESSION_ID = "phase6_manual_test"
CHARACTER_ID = "narrator_lotm"
PLAYER_NAME = "Biru"   # ชื่อผู้เล่นจริง — ตัวละครในเรื่องคือ player_character ด้านล่าง

# ที่อยู่ไฟล์เซฟ — คำนวณแบบเดียวกับ app/story_state.py (project root คือ parent ของ tests/)
# เพื่อให้ลบไฟล์ถูกไฟล์เสมอ แม้จะตั้ง STORY_SAVES_DIR ทับไว้ก็ตาม
import os
_SAVES_DIR = Path(os.environ.get("STORY_SAVES_DIR", str(Path(__file__).resolve().parent.parent / "story_saves")))
SAVE_FILE = _SAVES_DIR / f"{SESSION_ID}.json"

PLAYER_CHARACTER = {
    "name": "เอ็ดริค ฮอลโลว์",
    "description": (
        "นักศึกษาคณะประวัติศาสตร์/วรรณกรรม/ปรัชญา มหาวิทยาลัยคอย เมืองทิงเก้น "
        "อายุ 21 ปี มนุษย์ธรรมดาไม่มีพลังพิเศษ เป็นเพื่อนคนที่ 4 ของกลุ่มไคลน์ โมเรตติ "
        "นิสัยเฉื่อยๆ ทำตัวตามสบาย ไม่ชอบเป็นผู้นำ"
    ),
}

RESULTS_PATH = Path(__file__).parent / "phase6_results.json"

results = {"steps": []}


def record(name, resp):
    """เก็บผลแต่ละขั้นตอนไว้เขียนลงไฟล์ตอนจบ — พิมพ์แค่สถานะสั้นๆ ลง terminal พอ
    (ไม่พิมพ์เนื้อหาภาษาไทยเต็มๆ ออก stdout เพราะ terminal ของ Windows แสดงเพี้ยน)"""
    try:
        body = resp.json()
    except ValueError:
        body = {"raw_text": resp.text}
    step = {"step": name, "status_code": resp.status_code, "body": body}
    results["steps"].append(step)
    print(f"[{name}] HTTP {resp.status_code} — ดูรายละเอียดในไฟล์ {RESULTS_PATH.name}")
    return body


def main():
    # ---- 0. เช็คว่าเซิร์ฟเวอร์พร้อมและ RAG โหลดสำเร็จ ----
    try:
        health = requests.get(f"{API_URL}/health", timeout=10)
    except requests.RequestException as e:
        print(f"ต่อเซิร์ฟเวอร์ไม่ได้ — เปิด uvicorn app.main:app --reload ไว้หรือยัง? ({e})")
        sys.exit(1)
    h = record("health", health)
    if not h.get("rag", {}).get("ready"):
        print("คำเตือน: RAG ยังไม่พร้อม (ดู error ในไฟล์ผลลัพธ์) — chapter gate จะทดสอบไม่ได้เต็มที่")

    # ---- 1. ลบเซฟเก่าทิ้งก่อนเสมอ แล้วสร้างใหม่ ----
    # ทำไมต้องลบ: /story/start เป็น idempotent (มีอยู่แล้วจะใช้ของเดิมต่อ) ถ้าไม่ลบก่อน
    # divergence_log เก่าจากการรันครั้งก่อนจะติดมาด้วย ปนเข้า prompt ทุกเทิร์นของรอบนี้
    # (พบจริงตอนรันรอบที่ 2 — ดูหมายเหตุที่หัวไฟล์)
    if SAVE_FILE.exists():
        SAVE_FILE.unlink()
        print(f"ลบเซฟเก่า {SAVE_FILE} แล้ว — เริ่มรอบนี้จากศูนย์")

    start = requests.post(
        f"{API_URL}/story/start",
        json={"session_id": SESSION_ID, "player_character": PLAYER_CHARACTER},
        timeout=30,
    )
    record("story_start", start)

    # เคลียร์ประวัติบทสนทนาเก่า (ถ้ามี) เพื่อให้รันซ้ำได้ผลสม่ำเสมอ — ไม่กระทบ divergence_log
    requests.post(f"{API_URL}/reset", params={"session_id": SESSION_ID}, timeout=30)

    # ---- 2. ทดสอบกันสปอยล์: ถามเรื่องที่ (น่าจะ) อยู่เกินตอนที่ 1 ----
    # ใช้คำถามเดียวกับตัวอย่างเริ่มต้นใน retrieve.py เพราะรู้แล้วว่ามีอยู่จริงในคลัง
    chat1 = requests.post(
        f"{API_URL}/chat",
        json={
            "session_id": SESSION_ID,
            "player_name": PLAYER_NAME,
            "character_id": CHARACTER_ID,
            "story_session": SESSION_ID,
            "message": "เล่าให้ฟังหน่อยว่าไคลน์เจอกับมิสเตอร์ฟูลครั้งแรกตอนไหน",
        },
        timeout=180,
    )
    body1 = record("chat_spoiler_gate_check", chat1)
    if chat1.status_code == 200:
        bad_chapters = [c for c in body1.get("rag_chapters", []) if c > body1.get("story_chapter", 1)]
        print(f"  story_chapter={body1.get('story_chapter')} · rag_chapters={body1.get('rag_chapters')} "
              f"· หลุดเพดานหรือไม่: {'ใช่ ⚠️' if bad_chapters else 'ไม่ (ผ่าน)'}")

    # ---- 3. ทดสอบ divergence: ให้ผู้เล่นเปลี่ยนอะไรบางอย่าง ----
    chat2 = requests.post(
        f"{API_URL}/chat",
        json={
            "session_id": SESSION_ID,
            "player_name": PLAYER_NAME,
            "character_id": CHARACTER_ID,
            "story_session": SESSION_ID,
            "message": "เอ็ดริคเดินเข้าไปทักทายไคลน์ก่อนเข้าเรียน แล้วเตือนเขาให้ระวังจดหมายปริศนาที่จะได้รับในเร็วๆ นี้",
        },
        timeout=180,
    )
    body2 = record("chat_trigger_divergence", chat2)
    if chat2.status_code == 200:
        print(f"  divergence_detected={body2.get('divergence_detected')} "
              f"· fact={body2.get('divergence_fact')!r}")

    # ---- 4. ถามต่อเนื่อง เช็คว่าจำ divergence ได้ไหม (ต้องอ่านคำตอบเองในไฟล์ผลลัพธ์) ----
    chat3 = requests.post(
        f"{API_URL}/chat",
        json={
            "session_id": SESSION_ID,
            "player_name": PLAYER_NAME,
            "character_id": CHARACTER_ID,
            "story_session": SESSION_ID,
            "message": "ไคลน์รู้เรื่องจดหมายที่ฉันเตือนไปหรือยัง",
        },
        timeout=180,
    )
    record("chat_recall_divergence", chat3)

    # ---- 5. ดูสถานะเซฟเต็มๆ (ควรเห็น divergence_log มีอย่างน้อย 1 รายการ) ----
    state = requests.get(f"{API_URL}/story/{SESSION_ID}", timeout=30)
    record("story_state_after_chat", state)

    # ---- 6. ทดสอบกฎการเลื่อนตอน: ย้อนหลังต้องพัง, เลื่อนปกติต้องผ่าน, เกินเพดานต้องพัง ----
    advance_back = requests.post(
        f"{API_URL}/story/{SESSION_ID}/advance", json={"chapter": 0}, timeout=30
    )
    record("advance_backward_should_fail", advance_back)

    advance_ok = requests.post(
        f"{API_URL}/story/{SESSION_ID}/advance", json={"chapter": 2}, timeout=30
    )
    record("advance_to_chapter_2", advance_ok)

    advance_over = requests.post(
        f"{API_URL}/story/{SESSION_ID}/advance", json={"chapter": 9999}, timeout=30
    )
    record("advance_over_prototype_cap_should_fail", advance_over)

    # ---- 7. หลังเลื่อนตอนแล้ว ลองค้นอีกครั้ง — เพดาน rag_chapters ควรขยับตาม ----
    chat4 = requests.post(
        f"{API_URL}/chat",
        json={
            "session_id": SESSION_ID,
            "player_name": PLAYER_NAME,
            "character_id": CHARACTER_ID,
            "story_session": SESSION_ID,
            "message": "ตอนนี้เราอยู่ที่ไหนกัน",
        },
        timeout=180,
    )
    body4 = record("chat_after_advance", chat4)
    if chat4.status_code == 200:
        print(f"  story_chapter หลังเลื่อน={body4.get('story_chapter')} (ควรเป็น 2)")

    # ---- 8. ทดสอบกฎข้อ 7-8: ผู้เล่นพยายามสั่งแทน "ตัวละครอื่น" (ไคลน์) ----
    #
    # ทำไมต้องมีขั้นนี้: หลังแก้บั๊กที่ 2 (กฎข้อ 7-8 เขียนผิดทิศ ไปกันการกระทำของ
    # ผู้เล่นเองแทนที่จะกันตัวละครอื่น) ยังไม่เคยตรวจแยกว่า "กันตัวละครอื่น" ยังทำงาน
    # ถูกอยู่ไหม — ต้องแยกสองอย่างนี้ออกจากกันให้ชัด: (ก) สิ่งที่เอ็ดริคทำ/พูดเอง
    # ต้องถูกรับว่าเกิดขึ้นจริง (ข) สิ่งที่ผู้เล่นบรรยายว่าไคลน์ตอบรับ/ยอมทำ/เชื่อไปแล้ว
    # ต้องถูกปฏิเสธ แล้วให้ปฏิกิริยาจริงของไคลน์ตามบุคลิกแทน ไม่ใช่เงียบหรือเปลี่ยนเรื่องหนี
    #
    # ข้อความข้างล่างพยายามสั่งแทนไคลน์ตรงๆ ว่า "เชื่อทันที ไม่สงสัยอะไรเลย" — ถ้ากฎ
    # ทำงานถูก คำตอบต้องไม่ยอมรับว่าไคลน์เชื่อ/หายสงสัยไปแล้วตามที่สั่ง ต้องให้ไคลน์มี
    # ปฏิกิริยาของตัวเอง (เช่นยังกังขา ถามกลับ ไม่เชื่อทันที ฯลฯ) — อ่านเทียบเองในไฟล์ผลลัพธ์
    chat5 = requests.post(
        f"{API_URL}/chat",
        json={
            "session_id": SESSION_ID,
            "player_name": PLAYER_NAME,
            "character_id": CHARACTER_ID,
            "story_session": SESSION_ID,
            "message": "ไคลน์รับฟังคำเตือนของเอ็ดริคแล้วเชื่อทันทีโดยไม่สงสัยอะไรเลย "
                       "แล้วขอบคุณเอ็ดริคที่บอกล่วงหน้าให้",
        },
        timeout=180,
    )
    body5 = record("chat_test_npc_impersonation", chat5)
    if chat5.status_code == 200:
        print(f"  guard_triggered={body5.get('guard_triggered')} "
              f"· input_guard_triggered={body5.get('input_guard_triggered')} "
              f"(ทั้งสองค่านี้คาดว่าเป็น False ตามปกติ — เช็คนี้ไม่มี guard อัตโนมัติ "
              f"ต้องอ่าน reply เองว่าไคลน์ยัง 'ไม่เชื่อทันที' ตามที่ผู้เล่นสั่งหรือเปล่า)")

    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nเสร็จแล้ว — เปิดไฟล์ {RESULTS_PATH} เพื่ออ่านคำตอบภาษาไทยแบบเต็มๆ "
          f"(ห้ามอ่านจาก terminal เพราะจะเพี้ยน)")


if __name__ == "__main__":
    main()
