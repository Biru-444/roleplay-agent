"""
test_sillytavern.py — ทดสอบชั้นแปลภาษา OpenAI-compatible (เฟส 7) ก่อนต่อ SillyTavern จริง

ทำไมต้องมีสคริปต์นี้ก่อนเปิด SillyTavern จริง: ถ้าต่อ ST แล้วพังจะแยกไม่ออกว่าปัญหาอยู่ที่
ฝั่งเรา (backend) หรือฝั่ง ST เอง (ตั้งค่าผิด, เวอร์ชันไม่ตรงกัน) — รันสคริปต์นี้ผ่านก่อน แปลว่า
ฝั่งเรา "พูดภาษา OpenAI ถูก" แน่นอนแล้ว เหลือแค่ตั้งค่า ST ให้ชี้มาถูกที่

วิธีใช้ (เหมือน tests/benchmark.py และ tests/test_phase6.py):
    1. เปิดเซิร์ฟเวอร์ไว้ก่อน:  uvicorn app.main:app --reload
    2. อีกหน้าต่างหนึ่งรัน:      python tests/test_sillytavern.py

ผลทุกขั้นตอนเก็บไว้ที่ tests/sillytavern_results.json — เปิดไฟล์นี้อ่าน ไม่ใช่จาก stdout
(บทเรียนเดิม: terminal ของ Windows แสดงภาษาไทยเพี้ยน)

ทำไม session นี้ไม่ล้างทิ้งก่อนรันเหมือน test_phase6.py: ตรงข้ามกับเทสต์เฟส 6 ที่ต้องการ state
สะอาดทุกรอบเพื่อเทียบผลก่อน/หลังแก้บั๊ก สคริปต์นี้จำลองการใช้งานจริงผ่าน SillyTavern ซึ่งเป็น
บทสนทนาต่อเนื่องไปเรื่อยๆ ไม่มีการรีเซ็ตทุกครั้งที่เปิดโปรแกรม — รันซ้ำหลายรอบตั้งใจให้ต่อบท
สนทนาเดิม เพื่อดูว่า multi-turn ผ่าน endpoint นี้ยังทำงานต่อเนื่องถูกต้องเหมือนผ่าน /chat ตรงๆ ไหม

หลังรันสคริปต์นี้ผ่านแล้ว วิธีตั้งค่าใน SillyTavern จริง:
    1. API เลือก "Chat Completion"
    2. Chat Completion Source เลือก "Custom (OpenAI-compatible)"
    3. Custom Endpoint (Base URL) ใส่:  http://localhost:8000/v1
    4. API Key ใส่อะไรก็ได้ (backend นี้ไม่ตรวจ — ต้นแบบสำหรับใช้ในเครื่องตัวเองเท่านั้น)
    5. กด "Connect" — ควรเห็นชื่อโมเดล "roleplay-agent" ขึ้นในดรอปดาวน์ (มาจาก GET /v1/models)
    6. **สำคัญ: ปิดตัวเลือก "Streaming"** — เวอร์ชันนี้ยังไม่รองรับ (ดูเหตุผลใน app/sillytavern.py)
       ถ้าลืมปิดจะได้ error 400 ที่บอกวิธีแก้ตรงๆ อยู่แล้ว
    7. ตัวละคร/เนื้อเรื่องที่คุยด้วยคือ narrator_lotm ที่ผูกกับเซฟ "sillytavern_session" เสมอ
       (ค่าเริ่มต้น — เปลี่ยนได้ผ่าน env var ST_CHARACTER_ID / ST_STORY_SESSION_ID ถ้าต้องการ
       ดูรายชื่อ env var ทั้งหมดได้ในหัวไฟล์ app/sillytavern.py) การ์ดตัวละครที่สร้างในหน้า ST เอง
       จะไม่ถูกใช้เลย เพราะ backend คุม system prompt/ตัวละครเองทั้งหมด — ตกลงกันไว้แบบนี้เพราะ
       โปรเจกต์นี้ยังเป็นต้นแบบส่วนตัว ยังไม่ต้องรองรับหลายตัวละครพร้อมกันผ่าน ST
"""
import json
import sys
from pathlib import Path

import requests

API_URL = "http://localhost:8000"
RESULTS_PATH = Path(__file__).parent / "sillytavern_results.json"

results = {"steps": []}


def record(name, resp):
    try:
        body = resp.json()
    except ValueError:
        body = {"raw_text": resp.text}
    step = {"step": name, "status_code": resp.status_code, "body": body}
    results["steps"].append(step)
    print(f"[{name}] HTTP {resp.status_code} — ดูรายละเอียดในไฟล์ {RESULTS_PATH.name}")
    return body


def main():
    # ---- 0. เช็คว่าเซิร์ฟเวอร์พร้อม ----
    try:
        health = requests.get(f"{API_URL}/health", timeout=10)
    except requests.RequestException as e:
        print(f"ต่อเซิร์ฟเวอร์ไม่ได้ — เปิด uvicorn app.main:app --reload ไว้หรือยัง? ({e})")
        sys.exit(1)
    record("health", health)

    # ---- 1. GET /v1/models — ต้องมีอย่างน้อยหนึ่งโมเดลในลิสต์ ----
    # นี่คือ endpoint แรกที่ SillyTavern เรียกตอนกด "Connect" — ถ้าพังตรงนี้ ST จะไม่ให้คุยเลย
    models = requests.get(f"{API_URL}/v1/models", timeout=10)
    body_models = record("list_models", models)
    if models.status_code == 200:
        ids = [m.get("id") for m in body_models.get("data", [])]
        print(f"  โมเดลที่พบ: {ids} · ควรเห็นอย่างน้อยหนึ่งรายการ")

    # ---- 2. POST /v1/chat/completions พร้อม stream=true — ต้องพังแบบสุภาพ (400) ไม่ใช่ 500 ----
    stream_check = requests.post(
        f"{API_URL}/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "ทดสอบ"}],
            "stream": True,
        },
        timeout=30,
    )
    record("stream_should_fail_gracefully", stream_check)
    print(f"  stream=true ควรได้ 400 (ไม่ใช่ 500 หรือค้าง): {'ผ่าน ✅' if stream_check.status_code == 400 else 'ไม่ผ่าน ⚠️'}")

    # ---- 3. แชทจริงรอบแรก ผ่านรูปแบบ OpenAI (เลียนแบบสิ่งที่ SillyTavern จะส่งมาจริง) ----
    # แนบ system + ประวัติเก่าจำลองมาด้วย เพื่อยืนยันว่า backend เพิกเฉยส่วนนี้ถูกต้อง
    # (ดูเหตุผลใน _extract_last_user_message ของ app/sillytavern.py — ใช้แค่ข้อความ user ล่าสุด
    # ประวัติจริงคุมเองฝั่ง backend ผ่าน session_id คงที่ ไม่ใช่จาก messages ที่ ST ส่งมา)
    chat1 = requests.post(
        f"{API_URL}/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},  # ต้องถูกเพิกเฉย
                {"role": "user", "content": "สวัสดี ฉันชื่อเอ็ดริค"},
            ],
            "stream": False,
        },
        timeout=180,
    )
    body1 = record("chat_completions_turn_1", chat1)
    if chat1.status_code == 200:
        reply1 = body1.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"  reply เทิร์น 1 (ตัดมา 80 ตัวอักษรแรก): {reply1[:80]!r}")

    # ---- 4. แชทรอบสอง — เช็คว่าประวัติต่อเนื่องจริง (session_id คงที่ฝั่ง backend) ----
    chat2 = requests.post(
        f"{API_URL}/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "จำได้ไหมว่าฉันชื่ออะไร"}],
            "stream": False,
        },
        timeout=180,
    )
    body2 = record("chat_completions_turn_2_recall_check", chat2)
    if chat2.status_code == 200:
        reply2 = body2.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"  reply เทิร์น 2 — อ่านเองว่าจำชื่อ 'เอ็ดริค' ได้จากเทิร์นก่อนไหม: {reply2[:120]!r}")

    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nเสร็จแล้ว — เปิดไฟล์ {RESULTS_PATH} เพื่ออ่านคำตอบภาษาไทยแบบเต็มๆ "
          f"(ห้ามอ่านจาก terminal เพราะจะเพี้ยน)")


if __name__ == "__main__":
    main()
