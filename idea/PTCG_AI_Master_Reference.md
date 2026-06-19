# 🏆 Pokémon TCG AI Battle Challenge — เอกสารอ้างอิงฉบับสมบูรณ์ (Master Reference)

> เอกสารนี้รวบรวม **ทุกข้อมูลตั้งแต่ต้น** ของการวิเคราะห์การแข่งขัน "The Pokémon Company - PTCG AI Battle Challenge Simulation" บน Kaggle — ตั้งแต่ภาพรวม, รายละเอียดทางการ, dataset, กลยุทธ์, ไปจนถึงงานวิจัยเชิงลึกเรื่อง imperfect-information game AI และ roadmap สำหรับสร้างโมเดลเพื่อเป็นที่ 1
>
> **วันที่จัดทำ:** 2026-06-17
> **เป้าหมาย:** สร้าง AI agent ที่ชนะการแข่งขัน (อันดับ 1 บน ladder + report ที่ดีเพื่อรับเงินรางวัล)

---

## สารบัญ

- [ส่วนที่ 0: สรุปย่อสุด (TL;DR)](#ส่วนที่-0-สรุปย่อสุด-tldr)
- [ส่วนที่ 1: การแข่งขันนี้คืออะไร](#ส่วนที่-1-การแข่งขันนี้คืออะไร)
- [ส่วนที่ 2: รายละเอียดทางการของ Competition](#ส่วนที่-2-รายละเอียดทางการของ-competition)
- [ส่วนที่ 3: Dataset และ Card Data](#ส่วนที่-3-dataset-และ-card-data)
- [ส่วนที่ 4: กลยุทธ์ — edge จริงอยู่ที่ไหน](#ส่วนที่-4-กลยุทธ์--edge-จริงอยู่ที่ไหน)
- [ส่วนที่ 5: รายงานวิจัยเชิงลึก — Imperfect-Information Game AI](#ส่วนที่-5-รายงานวิจัยเชิงลึก--imperfect-information-game-ai)
- [ส่วนที่ 6: Roadmap สำหรับสร้างโมเดล](#ส่วนที่-6-roadmap-สำหรับสร้างโมเดล)
- [ส่วนที่ 7: Repos, Libraries และ Resources](#ส่วนที่-7-repos-libraries-และ-resources)
- [ส่วนที่ 8: ข้อควรระวัง (Caveats)](#ส่วนที่-8-ข้อควรระวัง-caveats)

---

## ส่วนที่ 0: สรุปย่อสุด (TL;DR)

**Pipeline ที่ชนะ:** imitation-bootstrap → self-play RL → inference-time belief search → deck meta-optimization

1. **Clone** replay ของ agent อันดับท็อปเป็น policy/value net (แนว DouZero/Suphx/AlphaStar)
2. **Refine** ด้วย league-style self-play + KL-anchoring เพื่อกัน strategic cycling
3. **เพิ่ม** Information-Set MCTS หรือ ReBeL-style belief resolving ตอนตัดสินใจ
4. **เลือก** deck ที่ robust (exploitability ต่ำ) ด้วย Nash mixture จาก matchup matrix

**บทเรียนสำคัญที่สุด:** คุณ **ไม่จำเป็นต้องมี CFR solver เต็มรูปแบบ** — model-free RL policy/value net ที่เทรนด้วย self-play (เสริม Nash regularization + cheap search) คือทางที่ใช้ได้จริงบนงบ compute/เวลาของ Kaggle

**⚠️ ข้อมูลที่ยังไม่ยืนยัน:** ชื่อ engine "cabt", ชื่อไฟล์ deck.csv/main.py, การ export replay รายวัน, time/GPU limit ต่อ turn — สิ่งเหล่านี้อยู่หลัง Kaggle rules wall **ข้อมูลที่ยืนยันแล้ว:** ส่งได้ 5 ครั้ง/วัน, งบเวลา 10 นาที/ผู้เล่น/แมตช์, การ์ด Standard ~2,000 ใบ, เป็น agent-vs-agent ladder

---

## ส่วนที่ 1: การแข่งขันนี้คืออะไร

### 🎯 ภาพรวม

การแข่งขันบน Kaggle ชื่อ **"PTCG AI Battle Challenge Simulation"** ซึ่งเป็นรูปแบบพิเศษที่ **AI ของผู้เข้าแข่งขันต่อสู้กันเอง** ไม่ใช่แค่ส่ง CSV ทำนายผลธรรมดา

จุดสำคัญ: นี่ไม่ใช่ **"การทำนาย score"** แต่เป็นการแข่ง **"สร้าง deck + สร้าง AI ต่อสู้"** ในรูปแบบ **ladder**

### 📋 สิ่งที่ต้องทำ (2 อย่าง)

| งาน | รายละเอียด |
|-----|-----------|
| **A. สร้างสำรับไพ่ (Deck)** | สำรับ 60 ใบ ส่งเป็น `deck.csv` ระบุการ์ดด้วย ID |
| **B. เขียน Play AI** | เขียนโค้ด `main.py` ให้ AI อ่านสถานการณ์แล้วตัดสินใจเล่น (ออกการ์ด / วาง bench / ใส่พลังงาน / วิวัฒนาการ / โจมตี / จบเทิร์น) |

### 🏆 ระบบการประเมิน (Ladder)

- AI ที่ส่งจะผ่าน validation (สู้กับตัวเอง) ก่อน แล้วเข้าสู่ **ระบบ Ladder อัตโนมัติ**
- จับคู่กับคู่ต่อสู้ที่ skill ใกล้เคียงกันซ้ำๆ
- ชนะ → Skill Rating **ขึ้น** / แพ้ → Skill Rating **ลง**
- เปรียบเหมือน **ลีกโปเกมอนการ์ดบน Kaggle**

### 💡 ความน่าสนใจ (นอกจากความแม่นยำของโมเดล)

- 🃏 การสร้างสำรับไพ่ (deck building)
- 🧠 การอ่านสถานการณ์บนบอร์ด (board judgment)
- 🎲 ความไม่แน่นอน (จั่วไพ่, โยนเหรียญ — random)
- 👁️ ข้อมูลไม่สมบูรณ์ (ไม่เห็นไพ่มือคู่ต่อสู้ — imperfect information)
- ⚔️ ความได้เปรียบด้านประเภทไพ่ (deck matchup / type matchup)
- 📊 การอ่าน Meta บน ladder (metagame)

> **"ความแข็งแกร่งไม่ได้ถูกตัดสินด้วยความแม่นยำของโมเดลเพียงอย่างเดียว"**

### 🃏 ผลทดสอบ Sample Rule-based Agent (~15,000 self-play)

| อันดับ | สำรับ | อัตราชนะ |
|--------|-------|---------|
| 🥇 1 | Mega Lucario ex | 60.4% |
| 🥈 2 | Dragapult ex | 55.6% |
| 🥉 3 | Iono | 43.8% |
| 4 | Mega Abomasnow ex | 40.2% |

**ข้อสังเกตสำคัญ:**
- Mega Lucario ex vs Dragapult ex → การปะทะกันตรงๆ เกือบ **50:50**
- Iono → win rate รวมต่ำ แต่ **ชนะ Mega Abomasnow ex ได้ค่อนข้างขาด**
- ⇒ **"เลือก deck ที่ win rate เฉลี่ยสูงสุด" ไม่ใช่คำตอบ** — ต้องดู **ความเข้ากัน (matchup) · แนวทางการเล่น · การอ่าน meta** ด้วย (มี rock-paper-scissors dynamics)

### 🚀 แนวทางพัฒนาสู่อันดับสูง (จาก infographic)

1. **rule-based** → เข้าร่วมก่อน
2. **วิเคราะห์ log** การแข่งขัน
3. **self-play** ฝึกเอง
4. **ค้นหา deck** ที่ดีกว่า (deck search)
5. **MCTS / Reinforcement Learning / Imitation Learning**

> 💡 เริ่มจาก simple ก็เข้าร่วมได้ แต่มีพื้นที่พัฒนา (伸びしろ) อีกมาก

---

## ส่วนที่ 2: รายละเอียดทางการของ Competition

### 📌 ข้อมูลพื้นฐาน

- ชื่อ: **The Pokémon Company - PTCG AI Battle Challenge Simulation**
- เป้าหมาย: สร้าง **AI Training Agent** เล่น Pokémon Trading Card Game
- โฟกัสงานวิจัย: เทรน AI สำหรับการเล่นแข่งขันในระบบที่ **ความน่าจะเป็น, องค์ประกอบที่ไม่รู้, การวางแผนเชิงกลยุทธ์** เป็นตัวตัดสินความสำเร็จ
- **มี 2 การแข่งขัน:** Simulation (อันนี้) + Hackathon/Strategy (การเข้าร่วม Hackathon ไม่บังคับสำหรับการเข้า Simulation)

### 🎮 ทำไม Pokémon TCG ถึงยากสำหรับ AI

- การ์ดจำนวนมาก, มี combination หลายพันแบบ
- ต้องคำนึงถึงกลยุทธ์/เด็ค/มือของคู่ต่อสู้
- มี Pokémon types หลายชนิด
- มีตัวแปรเพิ่มจาก **card draw และ coin toss**
- **ไม่รู้ว่าคู่ต่อสู้ถือการ์ดอะไร** = ความท้าทายหลัก
- **rule-based อย่างเดียวไม่พอ** ที่จะได้อันดับสูง — ต้อง forward thinking, real-time adaptation, optimal decision-making

### 🛠️ Simulator (SDK)

- ผู้เข้าแข่งขันได้รับ **simulator (SDK) สำหรับเทรนและทดสอบ**
- **ใช้ logic เดียวกับ environment ของ Kaggle** → เหมาะกับ local debugging และ reinforcement learning
- Battle รันบน **cabt Engine** (Pokémon TCG battle simulator สำหรับ kaggle-environments) *[ชื่อ engine ยังไม่ยืนยันจาก public source]*
- API docs: `https://matsuoinstitute.github.io/cabt/`
- **⚠️ มีข้อแตกต่างระหว่างกฎ TCG จริงกับพฤติกรรมของ simulator** (ดูเอกสาร differences)

**การทำงานของ engine:** แต่ละ turn agent ได้รับ **observation** (game logs + board state ปัจจุบัน + list ของ legal options) แล้ว return **index ของ option ที่เลือก** — engine จะแสดงเฉพาะ **legal moves** เท่านั้น

### ⚙️ Evaluation (สำคัญมาก)

- ส่งได้ **สูงสุด 5 agents/วัน/ทีม**
- แต่ละ submission เล่น **Episodes (games)** กับ agent อื่นบน ladder ที่ skill rating ใกล้กัน
- skill rating ขึ้นเมื่อชนะ / ลงเมื่อแพ้ / เฉลี่ยเมื่อเสมอ
- **ระบบ track เฉพาะ 2 submissions ล่าสุด** (เพื่อลดจำนวน agent และเพิ่มจำนวน episode)
- ทุก agent ที่ส่งจะเล่น episode ต่อไปเรื่อยๆ จนจบ competition — **agent ใหม่เล่นถี่กว่า**
- บน leaderboard แสดงเฉพาะ agent ที่คะแนนดีที่สุด แต่ track ทุกตัวได้ในหน้า Submissions

### 📊 Ranking System (TrueSkill-style)

- แต่ละ submission มี **Skill Rating** model ด้วย **Gaussian N(μ, σ²)**
  - **μ** = skill ที่ประมาณ (เริ่มต้น μ₀ = **600**)
  - **σ** = ความไม่แน่นอนของการประมาณ (ลดลงเมื่อเวลาผ่านไป)
- เมื่อ upload → เล่น **Validation Episode** กับตัวเองก่อน (เช็คว่าทำงานได้)
  - ถ้า fail → mark เป็น **Error** + download agent logs ได้
  - ถ้าผ่าน → initialize ที่ μ₀=600 เข้า pool
- หลังจบ Episode → update rating ทุก submission ในเกมนั้น
  - ผู้ชนะ μ เพิ่ม / ผู้แพ้ μ ลด / เสมอ → ขยับ μ เข้าหา mean
  - ขนาดการ update สัมพันธ์กับความเบี่ยงเบนจากผลที่คาดไว้ และกับ σ ของแต่ละ submission
  - σ ลดลงตามปริมาณข้อมูลที่ได้
- **🔑 คะแนนที่ชนะ/แพ้ (margin) ไม่มีผลต่อ skill rating update** — มีแค่ ชนะ/แพ้/เสมอ

### 📅 Timeline

| วันที่ | เหตุการณ์ |
|--------|----------|
| **16 มิ.ย. 2026** 11:00 UTC | Start Date |
| **9 ส.ค. 2026** | Entry Deadline (ต้องยอมรับ rules ก่อนวันนี้) |
| **9 ส.ค. 2026** | Team Merger Deadline |
| **16 ส.ค. 2026** | Final Submission Deadline (ล็อค submission) |
| **17–~31 ส.ค. 2026** | รันเกมต่อจน leaderboard converge → leaderboard final |

หมายเหตุ: deadline ทั้งหมด 11:59 PM UTC เว้นแต่ระบุไว้เป็นอื่น

### 💰 Prizes (สำคัญต่อเป้า "ที่ 1")

- **Competition track (Simulation) เอง ไม่มีเงินรางวัล**
- เงินอยู่ที่ **Hackathon/Strategy track** — ต้องส่ง **report** อธิบาย strategic logic
- โครงสร้างรางวัล (จากแหล่งข่าว):
  - **top 8 ทีม** จาก Strategy Category → ทีมละ **$30,000** เข้ารอบ Final
  - **Final Stage** กันยายน 2026 ที่ญี่ปุ่น (แข่งสด, stream บน YouTube)
  - ที่ 1: **$50,000** / ที่ 2: **$30,000**
  - + Google Cloud credits ($3k/participant ตามรายงานบางแหล่ง)
- final ranking ของ Hackathon ตัดสินจาก **Competition leaderboard + Hackathon evaluation**

> **⚠️ ต่อให้ขึ้นอันดับ 1 บน ladder แต่ถ้าไม่เขียน report ดี ก็ไม่ได้เงิน → ต้องทำทั้งสองอย่างควบกัน**

### 📤 วิธี Submit

- ต้องเป็น **.tar.gz bundle** ที่มี **main.py** ที่ top level (ไม่ nested) และมี **deck.csv**
- สร้างด้วย: `tar -czvf submission.tar.gz *`
- upload ในแท็บ My Submissions
- submission จะเริ่มด้วยเกมกับตัวเองเพื่อเช็คก่อนเข้า matchmaking pool
- อ้างอิง config: kaggle-environments 1.14.10 (`github.com/Kaggle/kaggle-environments`)

### 👥 ผู้จัดและคู่แข่ง

- จัดโดย **The Pokémon Company** + **HEROZ** + **Matsuo Institute**
- หนุนหลังโดย **Google, Google Cloud, NVIDIA, Kaggle**
- **คู่แข่งระดับสูง:** HEROZ (บริษัท AI ที่ทำ AI หมากรุกญี่ปุ่นระดับชนะแชมป์), Matsuo Institute (แล็บ AI อันดับต้นของญี่ปุ่น) → **rule-based ธรรมดาแพ้แน่นอน**
- เปิดให้ทุกคน (ยกเว้นบางภูมิภาคที่ถูก sanction เช่น เกาหลีเหนือ)

### Citation

The Pokémon Company, HEROZ, Matsuo Institute, Addison Howard, Bovard Doerschuk-Tiberi. *The Pokémon Company - PTCG AI Battle Challenge Simulation.* https://kaggle.com/competitions/pokemon-tcg-ai-battle, 2026. Kaggle.

---

## ส่วนที่ 3: Dataset และ Card Data

### 🎬 Episode Replay

- เข้าถึง replay ของ submission ตัวเองได้จากแท็บ **Submissions** หรือ download ผ่าน CLI/MCP
  - docs: `github.com/Kaggle/kaggle-cli/blob/main/docs/simulation_competitions.md`
- download replay ของทีมอื่นได้จาก **Leaderboard**
- 🔑 ทาง Kaggle จะ **export episode รายวันของอันดับท็อป** (เพื่อช่วยทำ BC/RL/IL) — โพสต์ใน competition forums
  - *[การ export รายวันยังไม่ยืนยัน 100% จาก public source แต่สอดคล้องกับ kaggle-environments substrate]*

### 📁 ไฟล์ที่ได้รับ

| ไฟล์ | รายละเอียด |
|------|-----------|
| `Card_ID_List_EN.pdf` | รายการการ์ดทั้งหมด (ภาษาอังกฤษ) พร้อม card ID, ชื่อ, expansion, collection number, รูปภาพ |
| `Card_ID_List_JP.pdf` | รายการการ์ดทั้งหมด (ภาษาญี่ปุ่น) โครงสร้างเหมือน EN |
| `EN Card Data.csv` | metadata การ์ดทุกใบ (ภาษาอังกฤษ) |
| `JP Card Data.csv` | metadata การ์ดทุกใบ (ภาษาญี่ปุ่น) |

> เนื้อหา EN/JP เหมือนกัน ต่างแค่ภาษาของชื่อ/คำอธิบายการ์ด

### 🗂️ โครงสร้าง CSV (Schema)

ทั้งสองไฟล์ใช้ schema เดียวกัน:

| Field | คำอธิบาย |
|-------|----------|
| **Card ID** | ตัวระบุเฉพาะที่ simulator ใช้ (🔑 สำคัญสุดสำหรับ deck.csv และโค้ด) |
| **Card Name** | ชื่อการ์ด |
| **Expansion** | ชุด expansion ที่สังกัด |
| **Collection No.** | หมายเลขใน expansion |
| **Stage / Type** | ระดับวิวัฒนาการ (Basic/Stage 1/Stage 2) หรือประเภทของ Energy/Trainer |
| **Rule** | ข้อความกฎพิเศษ (ถ้ามี — เช่น ex Rule) |
| **Category** | ประเภทการ์ด (Pokémon / Trainer / Energy) |
| **Previous stage** | ระดับวิวัฒนาการก่อนหน้าที่ต้องใช้ |
| **HP** | Hit Points |
| **Type** | ประเภทโปเกมอน (Grass, Fire, Water, ...) |
| **Weakness** | ประเภทที่อ่อนแอต่อ |
| **Resistance (Type)** | ประเภทที่ต้านทาน |
| **Retreat** | ค่า retreat เพื่อสลับโปเกมอน |
| **Move Name** | ชื่อท่าโจมตี/move |
| **Cost** | พลังงานที่ต้องใช้ |
| **Damage** | ความเสียหาย |
| **Effect Explanation** | คำอธิบาย effect ของท่า/กฎเพิ่มเติม |

### 💡 นำ Card Data ไปใช้อะไร

```
Card Data CSV
    ├── สร้าง deck.csv → เลือกการ์ด 60 ใบ โดยใช้ Card ID
    ├── วิเคราะห์ Synergy ของการ์ด
    ├── คำนวณ Weakness/Resistance เพื่อให้ AI อ่านเกมได้
    ├── ทำความเข้าใจ Effect ของแต่ละท่า → เขียน Logic / encode ใน network
    └── สร้าง feature encoding (card matrix) สำหรับ policy/value net
```

> **🔑 Card ID คือสิ่งสำคัญที่สุด** — ต้องใช้ตรงกับที่ Simulator กำหนด ทั้งใน deck.csv และในโค้ด

> **⚠️ อย่า scrape ข้อมูลการ์ด/meta จากเว็บนอก (limitlesstcg ฯลฯ) มาใช้ตรงๆ** เพราะ simulator ≠ กฎจริง → ground truth คือ engine เอง ไม่ใช่เกมจริง

---

## ส่วนที่ 4: กลยุทธ์ — edge จริงอยู่ที่ไหน

### ⚠️ จุดที่คนเข้าใจผิด: "scrape skill card" ไม่ใช่ทางชนะ

- ข้อมูลการ์ดทั้งหมด (HP, ท่า, Cost, Damage, Effect, Weakness/Resistance) **เขาแจกให้ครบแล้ว** → scrape เพิ่มไม่ได้เปรียบ
- scrape meta โลกจริงมาใช้ตรงๆ **อาจทำให้เข้าใจผิด** เพราะ simulator มีข้อแตกต่างจากกฎจริง
- **edge จริง = ขุดข้อมูลจากตัว engine เอง** (self-play + replay analysis) ไม่ใช่จากเว็บนอก

### 🎯 edge จริงอยู่ที่ไหน (3 ชั้น)

#### ชั้นที่ 1: Imitation Learning จาก replay ของทีมท็อป (ของฟรีที่เขายื่นให้)
- Kaggle export episode ของอันดับท็อปทุกวัน + download replay ทีมอื่นได้
- = ข้อมูลการเล่นของผู้เล่นเก่งที่สุดแบบ **supervised** ฟรีๆ
- train policy network เลียนแบบการเลือก move ของ agent อันดับสูง = ทางลัดไป baseline ที่แข็งเร็วที่สุด
- (ทีมที่ 19 ของ Lux AI ใช้ imitation agent เลียนแบบผู้เล่นเก่งจาก match record ผ่าน Kaggle API)

#### ชั้นที่ 2: Self-play RL (ตัวยกระดับ)
- SDK ใช้ logic เดียวกับ environment จริง → รัน self-play มหาศาลในเครื่องตัวเองได้
- บทเรียนจากผู้ชนะ Lux AI 2021:
  - ใช้ **IMPALA + UPGO + TD-lambda loss**
  - มี **frozen teacher model + KL loss** เพื่อกัน **strategic cycle** (ปัญหาคลาสสิกของ pure self-play)
  - **reward shaping** ช่วงต้น (ให้แต้มตอน KO / เก็บ prize card / ได้เปรียบบอร์ด) ก่อนเปลี่ยนเป็น sparse win/loss
- เคล็ดลับ "teacher KL" สำคัญมาก — กันการวนลูปกลยุทธ์เดิม

#### ชั้นที่ 3: Search ตอน inference + Belief modeling (ตัวแยกที่ 1 ออกจากที่ 10)
- ปัญหาหลัก: **ไม่เห็นมือคู่ต่อสู้** = โจทย์เดียวกับ Poker/Stratego
- **Determinized MCTS / IS-MCTS**: สุ่มมือ+เด็คที่คู่ต่อสู้น่าจะมี (ให้สอดคล้องกับการ์ดที่เห็นเขาเล่น) → รัน MCTS หลาย determinization → รวมผล
- **Belief / opponent modeling**: card counting — ติดตาม distribution ของมือคู่ต่อสู้ + เดา archetype จากการ์ดที่เขาโชว์
- เอา policy/value net จากชั้น 1-2 มาเป็นตัวประเมินใน MCTS แบบ AlphaZero (ดัดแปลงสำหรับ imperfect info)

### 🃏 deck.csv คือเกมที่สอง (Meta-game)

- ผลในภาพ (Lucario 60.4% ฯลฯ) เป็นแค่ผลของ sample agent — deck ที่ดีที่สุดต้อง **co-optimize กับ play AI ของคุณเอง**
- วิธีทำ:
  1. ให้ agent ที่แข็งสุด pilot deck หลายแบบสู้กันเอง → สร้าง **matchup matrix**
  2. อย่าเลือก win rate เฉลี่ยสูงสุด — เลือก deck ที่ **robust** (worst-case win rate สูง / exploitability ต่ำ)
  3. **meta ของ ladder เลื่อนตลอด 2 เดือน** → เตรียมเปลี่ยน deck ตามที่เจอบ่อยขึ้น

### ⚙️ กลยุทธ์เฉพาะของ Kaggle (อย่าทิ้งแต้มฟรี)

- ส่งได้ **วันละ 5 ครั้ง นับแค่ 2 ตัวล่าสุด** → **test ในเครื่องให้สุดก่อนส่ง** อย่าเผา submission
- **margin ที่ชนะไม่มีผลต่อ rating** → optimize เพื่อ "ความน่าจะเป็นที่จะชนะ" ห้ามเสี่ยงไลน์เพื่อชนะขาด
- agent ใหม่ได้เล่น episode ถี่กว่า (σ ลดเร็ว) → จังหวะการส่งมีผล
- **Validation = สู้กับตัวเอง (mirror match)** → agent ต้องไม่พังในกระจกเงา
- อ่านเอกสาร **"rule differences" ให้ละเอียด** — quirk ของ engine คือ edge ที่ผู้เล่นเกมจริงไม่รู้
- leaderboard มักจัดอันดับจาก **μ−kσ (conservative estimate)** → เมื่อ μ สูงแล้ว ให้ส่งต่อเพื่อกด σ ลง

---

## ส่วนที่ 5: รายงานวิจัยเชิงลึก — Imperfect-Information Game AI

> ส่วนนี้คือรายงานวิจัยฉบับเต็ม สังเคราะห์เทคนิคจากเกม imperfect-information ทั้งหมด (Poker, Stratego, Mahjong, DouDizhu, card games) แล้ว map กลับมา PTCG

### 5.0 Key Findings (สรุปผลวิจัย)

1. **CFR-family solvers ดีทางทฤษฎีแต่หนักในทางปฏิบัติ** — Vanilla CFR converge เป็น Nash ที่ O(1/√T); CFR+, Linear CFR, Discounted CFR converge เร็วกว่ามาก; MCCFR sample tree; Deep CFR ใช้ neural net แทน abstraction สำหรับ agent ที่จำกัดเวลาต่อ turn, full CFR ไม่ practical แต่ *ไอเดีย* (regret matching, counterfactual values, depth-limited resolving) เป็นรากฐานของทุกอย่าง

2. **State of the art = RL + search + belief states** — DeepStack/ReBeL ทำงานบน *public belief states* + depth-limited continual resolving ด้วย value network; Player of Games/Student of Games รวมเข้ากับ AlphaZero-style search รองรับทั้ง perfect + imperfect info; DeepNash แสดงทาง model-free RL (R-NaD) ถึง ε-Nash ในเกมใหญ่กว่า poker มาก (Stratego) **โดยไม่ต้อง search**

3. **Card-game AI ที่ชนะจริงใช้สูตรเรียบง่ายอย่างน่าประหลาด** — DouZero (Deep Monte-Carlo + action encoding + parallel actors) อันดับ 1 จาก 344 bots ด้วย Elo 1625.11; Suphx (supervised init → self-play RL + oracle guiding + global reward prediction) ถึง 8.74 dan บน Tenhou สูงกว่า 99.99% ของผู้เล่นมนุษย์ที่มี rank ทางการ; LOCM winners ใช้ search (MCTS, Rolling Horizon Evolution) และ end-to-end RL + fictitious play

4. **Determinized/IS-MCTS ใช้ได้แต่มี strategy fusion + non-locality** — fix ด้วย information-set trees, multi-observer IS-MCTS, particle-filter re-determinization, belief/opponent inference **IS-MCTS เคยชนะ determinized MCTS บน Pokémon (เกมต่อสู้) แล้ว** (IEEE SMC 2018: 57.5% vs 42.5%)

5. **การเลือก deck เป็นเกมในตัวมันเอง** — matchup matrix ระหว่าง deck = symmetric zero-sum metagame; Nash mixture = deck-selection ที่ unexploitable; co-optimize deck กับ play policy ด้วย quality-diversity (MAP-Elites) หรือ evolutionary search

6. **Kaggle ladder ในอดีต (Lux, Hungry Geese) ชนะด้วย imitation + distributed self-play RL** (HandyRL/IMPALA/PPO, V-trace/UPGO) + light MCTS ตอน inference — template ที่ transfer ได้ตรง เพราะรันบน kaggle-environments เดียวกัน

---

### 5.1 Core game-theoretic / search algorithms

#### Counterfactual Regret Minimization (CFR) และ variants

**CFR** (Zinkevich, Johanson, Bowling, Piccione, NIPS 2007) — algorithm no-regret พื้นฐานสำหรับ extensive-form imperfect-info games แตก total regret เป็น per-information-set *counterfactual regrets*, ใช้ *regret matching* แบบ local, **average** strategy converge เป็น Nash ใน 2-player zero-sum ที่ bound O(1/√T)

Variants สำคัญ:
- **CFR+** (Tammelin, 2014) — clip negative cumulative regret เป็น 0 ทุก iteration + linear averaging; เร็วกว่ามาก ใช้แก้ heads-up limit hold'em ได้ (Bowling et al., Science 2015)
- **Linear CFR / Discounted CFR (DCFR)** (Brown & Sandholm, AAAI 2019) — ถ่วงน้ำหนัก iteration เพื่อ discount ความผิดพลาดช่วงต้น; เร็วสุดในกลุ่ม tabular; DCFR+/Predictive CFR+ (Farina et al.) ไปไกลกว่า
- **Monte Carlo CFR (MCCFR)** (Lanctot et al., NIPS 2009) — sample tree (outcome-/external-sampling) แต่ละ iteration ถูก ไม่ต้อง traverse เต็ม — version ที่ใช้ได้โดยไม่ต้องมี perfect model
- **Deep CFR** (Brown, Lerer, Gross, Sandholm, ICML 2019, arXiv:1811.00164) — แทน abstraction ด้วย neural net ที่ approximate regret; **Single Deep CFR** (Steinberger, arXiv:1901.07621) ตัด average-strategy network ออก; งานใหม่ (arXiv:2511.08174) รวม advanced tabular variants เข้า neural CFR

**Map to PTCG:** tree ของ PTCG ใหญ่เกินไปสำหรับ tabular CFR; ใช้เชิงแนวคิด (regret-matching เป็น local policy update, counterfactual values สำหรับ resolving) ถ้ามี subgame เล็กที่ abstract ดี (เช่น endgame ใกล้จบ) → neural-CFR resolve ช่วยได้ แต่เป็น refinement ระยะหลัง

#### Poker milestones

- **Libratus** (Brown & Sandholm, Science 2018; IJCAI-17) — ชนะ pro heads-up no-limit: blueprint ผ่าน MCCFR over abstraction + nested safe **subgame solving** real-time + self-improver ที่ปะรูที่คู่ต่อสู้เจอ ใช้ ~**25 ล้าน core hours** บน Bridges supercomputer (PSC) + ~100 CPU live
- **Pluribus** (Brown & Sandholm, Science 2019) — ชนะ pro elite ที่ **6-player** no-limit (landmark เพราะ Nash ไม่ unique/ไม่การันตีชนะใน >2 ผู้เล่น) คำนวณ blueprint ด้วย **Monte Carlo Linear CFR** ใน 8 วัน 12,400 core-hours แล้ว **depth-limited search** ตอน test ที่ leaf ให้แต่ละผู้เล่นเลือกจาก population เล็กๆ ของ continuation/blueprint strategies — รัน live บนแค่ **2 CPU / <128GB RAM** → "cheap search ด้วย continuation strategy ไม่กี่ตัว" relevant มากกับ agent ที่ CPU-bound
- **DeepStack** (Moravčík et al., Science 2017) — **continual re-solving** บน public tree: maintain *range* ตัวเอง + vector ของ opponent counterfactual values, re-solve depth-limited subgame ทุก decision, แทน subtree ลึกด้วย learned **deep counterfactual value network** → ทำให้ depth-limited search ภายใต้ imperfect info sound
- **ReBeL** (Brown, Bakhtin, Lerer, Gong, NeurIPS 2020, arXiv:2007.13544) — generalize AlphaZero ไป imperfect info โดยใช้ **public belief state (PBS)** เป็น "state", train value+policy net over PBSs ด้วย self-play, รัน CFR-based search ทั้ง train/test converge เป็น Nash ใน 2p0-sum; reduce เป็น AlphaZero ใน perfect-info; FAIR open-source Liar's Dice code
- **Player of Games / Student of Games** (Schmid, Moravčík, Bard, Lanctot, Bowling et al.; Science Advances 2023) — รวม guided search + self-play + game-theoretic reasoning ผ่าน **Growing-Tree CFR (GT-CFR)** ชนะ Slumbot (poker), amateur Go, ชนะ SOTA Scotland Yard bot → **"general recipe" ที่ใกล้ PTCG ที่สุด** (ผสม perfect-info board + hidden hand/deck)

#### IS-MCTS / determinized MCTS (PIMC) และจุดอ่อน

**PIMC** sample ("determinize") hidden info เป็นโลก concrete, solve แต่ละโลกแบบ perfect-info, แล้วเฉลี่ย — simple ใช้ได้ดีใน Bridge/Skat แต่มี 2 pathology (Long et al., AAAI 2010):
- **Strategy fusion** — search ตัดสินใจต่างกันในแต่ละโลกที่ determinize ทั้งที่อยู่ใน information set เดียวต้อง commit กลยุทธ์เดียว
- **Non-locality** — optimal payoff ไม่ recursive ตาม subgame เพราะ hidden state ของคู่ต่อสู้ขึ้นกับ history

**Information Set MCTS (ISMCTS)** (Cowling, Powley, Whitehouse, IEEE TCIAIG 2012) — สร้าง tree over *information sets*, pool statistics ลด strategy fusion; **Multi-Observer ISMCTS** สร้าง tree ต่อผู้เล่น + opponent modeling เบื้องต้น; **Re-determinizing ISMCTS** + particle filter จัดการ non-locality; **Online Outcome Sampling** (MCCFR-based) ลด exploitability เมื่อ search มากขึ้น

🔑 **IS-MCTS เคยชนะ determinized MCTS บน Pokémon** (Ihara, Imai, Oyama, Kurihara, IEEE SMC 2018, DOI 10.1109/SMC.2018.00375; ISMCTS 57.5% vs DMCTS 42.5%) — หลักฐานตรงว่า family นี้ transfer ได้ (แต่เป็นเกมต่อสู้ ไม่ใช่ TCG)

**Map to PTCG:** ISMCTS คือ search ตอน inference ที่เป็นธรรมชาติที่สุด — determinize มือ+เด็คซ่อนของคู่ต่อสู้ (sample จาก belief model/card-counting), search information-set tree ในงบเวลาต่อ turn, ลด fusion ด้วย information-set pooling + re-determinization

#### DeepNash / R-NaD (Stratego)

DeepNash (Perolat et al., Science 2022, arXiv:2206.15378) — ถึง expert level ใน Stratego (tree ~10^535) **โดยไม่ใช้ search** ด้วย **Regularized Nash Dynamics (R-NaD)**: model-free RL ที่เพิ่ม regularization ให้ learning dynamics converge *ไปยัง* ε-Nash แทนที่จะ cycle ใช้ reward-transformation/dynamics/update loop + **v-trace** value estimator + **Neural Replicator Dynamics (NeuRD)** ชนะ 97% vs bots, 84% vs human experts

**บทเรียน PTCG:** RL policy ที่ search-free + equilibrium-regularized ใช้ได้และ unexploitable — น่าสนใจถ้า compute ต่อ turn ตึงเกินไปสำหรับ heavy search

#### Depth-limited & safe subgame solving

หลักการทั่วไป: ภายใต้ imperfect info จะ swap subtree เป็น heuristic value คงที่ไม่ได้ เพราะ leaf value ขึ้นกับ range ของทั้งสองฝ่าย; safe subgame solving สร้าง *augmented* game ที่การันตีว่า resolved strategy ไม่เพิ่ม exploitability; depth-limited ใช้ value network (DeepStack/ReBeL) หรือ population ของ continuation strategies ที่ leaf (Pluribus)

---

### 5.2 Card-game AI โดยเฉพาะ

- **DouZero** (Zha et al., ICML 2021, arXiv:2106.06135; `github.com/kwai/DouZero`): **Deep Monte-Carlo (DMC)** + **action-encoding** (การ์ดเป็น 4×15 one-hot matrix ทั้ง state และ action → generalize over action set ใหญ่ที่เปลี่ยนทุก turn) + **parallel actors** เทรนจาก scratch บน server เดียว 4 GPU (1080 Ti) + 48 processors, แซง supervised baseline ใน ~2 วัน, **อันดับ 1 จาก 344 bots บน Botzone ด้วย Elo 1625.11** (30 ต.ค. 2020) → *classic MC + good encoding + parallelism ชนะ MCTS/CFR ที่หนักกว่า* = **precedent ที่ apply กับ PTCG ตรงที่สุด** (action space combinatorial)

- **Suphx** (Li et al., Microsoft Research, arXiv:2003.13590): **5 residual-CNN models** (discard/Riichi/Chi/Pon/Kan, 100+ layers) เทรน **supervised จาก (state, action) ของผู้เล่นมนุษย์ระดับท็อปบน Tenhou** แล้ว **boost ด้วย self-play RL** + 3 นวัตกรรม: **global reward prediction** (credit assignment ข้าม hand/round), **oracle guiding** (ใช้ hidden info ตอนเทรน แล้วค่อยถอด), **run-time policy adaptation** → สูงกว่า 99.99% ผู้เล่นมนุษย์, **8.74 dan stable** (สถิติ 10 dan, AI ตัวแรก) → *imitation init + self-play RL + oracle ที่ใช้ hidden info เฉพาะตอนเทรน* = สูตร hidden-info ที่พิสูจน์แล้ว

- **LOCM / Strategy Card Game AI Competition** (Kowalski & Miernik; arXiv:2305.11814): benchmark CCG วิชาการเดียวที่มี **deckbuilding รวมในสนาม** (effect deterministic; nondeterminism จาก draw order + เด็คคู่ต่อสู้ที่ไม่รู้) winners ใช้ MCTS, **Rolling Horizon Evolution**, Pruned BFS, Dynamic Lookahead + state-eval; **end-to-end RL ครองทีหลัง** (Xi et al., arXiv:2303.04096 — เรียน draft + battle พร้อมกัน); Vieira et al. เทรน **self-play DRL drafter** ดันอันดับจาก 10 → 4 → **analogue ที่ใกล้ deck.csv + play split ของ PTCG ที่สุด**

- **Magic: The Gathering** — Cowling et al. ใช้ determinized MCTS เหนือ rule-based heuristic; **Hearthstone AI Competition** (2018–2020, SabberStone ~98% ของ base cards) มี Premade-Deck + User-Created-Deck track; winners ผสม MCTS + supervised state-eval + evolutionary; **Tales of Tribute** (arXiv:2305.08234) คือ successor deckbuilding-CCG competition

- **การจัดการส่วนยาก:**
  - action space ใหญ่/แปรผัน → action *encoding* + masking (DouZero)
  - มือคู่ต่อสู้ซ่อน → belief/range modeling (DeepStack ranges, Pluribus Bayesian range) + oracle guiding (Suphx)
  - draw stochasticity → Monte-Carlo value + determinization
  - long-horizon/sparse reward → global reward prediction (Suphx), reward shaping, value bootstrapping
  - evolution-stage sequencing → multi-step macro-actions / auto-regressive policy heads

- **Belief modeling / card counting:** track การ์ดที่เปิดเผย (discard, prize ที่เก็บ, bench, ขนาดมือ, การ์ดที่ search) → maintain posterior over มือซ่อน + deck order; sample determinization สำหรับ ISMCTS; optionally train "opponent hand predictor" network แยก

---

### 5.3 Self-play RL & Imitation Learning (transferable to kaggle-environments)

- **AlphaZero/MuZero adaptation:** AlphaZero = MCTS + self-play policy/value net (perfect info); **MuZero** เรียน dynamics model + จัดการ stochastic/partially observable Atari; **Stochastic MuZero** เพิ่ม chance nodes; **AlphaZe\*** (Frontiers in AI 2023) เก็บ machinery ของ AlphaZero แต่ swap MCTS เป็น PIMC-style planner — "surprisingly strong" บน Stratego/Dark Hex = middle ground

- **Algorithms ที่ชนะ Kaggle ladders:** **IMPALA + V-trace** (off-policy correction สำหรับ distributed actors), **PPO**, **UPGO** (จาก AlphaStar), **TD(λ)**, **league training** Lux AI S1/S2 top teams ใช้ deep RL + **imitation จาก downloaded episodes** เพื่อ bootstrap; Hungry Geese top solutions ใช้ **DeNA's HandyRL** + **UPGO→V-trace** + supervised imitation จาก LB>1200 episodes แล้ว **MCTS ตอน inference** ("AlphaGeese")

- **League training / anti-cycling (AlphaStar, Vinyals et al., Nature 2019):** *main agent* (prioritized fictitious self-play, PFSP) + *main exploiters* + *league exploiters* กัน rock-paper-scissors cycling; KL-regularize ไป frozen supervised "teacher" รักษาความหลากหลาย R-NaD (DeepNash) = ทางเลือก game-theoretic แทน league

- **Behavioral cloning จาก replays:** parse exported episode → (state, legal-action-mask, chosen-action, outcome) → train policy net ด้วย cross-entropy (+ value head จากผลเกม) ทั้ง init RL ด้วย prior ที่แข็ง + inject กลยุทธ์หลากหลายที่ pure-from-scratch RL หายาก (AlphaStar ~10^6 replays); iterative BC (re-clone จาก MCTS agent ที่ดีขึ้น + meta ล่าสุด) ใช้ใน Hungry Geese

- **State/action encoding architectures:**
  - **set-based / matrix encoders** (DouZero 4×15 card matrix; LOCM 120×16 card matrix; per-card feature rows สำหรับมือ/บอร์ดแปรผัน)
  - **ResNet/CNN** over spatial board (Hungry Geese: 8-layer ~46-channel ResNet dual policy/value)
  - **transformers/Deep Sets** สำหรับ permutation-invariant variable-size hands/benches
  - ใช้ **action masking** เสมอ; **auto-regressive policy heads** สำหรับ compound turns (เลือกการ์ด → เลือก target → ...)

---

### 5.4 Deck building / Meta optimization

- **Automated deckbuilding:** evolutionary algorithms (García-Sánchez et al. "Evolutionary deckbuilding in Hearthstone"; Kowalski & Miernik "Active Genes" arXiv:2001.01326), genetic algorithms (MtG), Q-DeckRec, **self-play DRL drafters** (gym-locm) **MAP-Elites / quality-diversity** เก็บ archive ของ deck เก่งหลากหลายใน feature space (aggro↔control, energy curve) → ได้ portfolio หลากหลาย valuable เมื่อ meta เลื่อนรายวัน

- **Nash-of-decks / metagame reasoning:** สร้าง **matchup matrix** M โดย M[i,j] = win-rate ของ deck i vs deck j = symmetric zero-sum game **Nash mixture** = distribution over decks ที่ unexploitable (ทุก deck ใน support มี 50% expected win-rate vs field) Wizards' "Metagame Mentor" คำนวณ equilibrium แบบนี้จริง (7-deck mixture ที่แต่ละ deck มี 50% expected win) เลือก Nash mixture → minimize worst-case exploitability; เลือก **best-response (maximally exploitative)** → ชนะมากกว่า *ถ้า*ทำนาย field ได้ แต่แพ้ counter-pick **บน ladder ที่ field กว้าง/เลื่อน → เอียงไป Nash deck (low-exploitability)**, shade ไป exploitative เฉพาะเมื่อ meta นิ่งและอ่านได้

- **Co-optimization:** deck value ขึ้นกับ play policy และกลับกัน → co-evolve: **PSRO / double-oracle** (เพิ่ม best-response policy+deck เข้า population แล้ว re-solve meta-Nash) เป็น framework หลัก; ถูกกว่าคือ alternating (fix policy → optimize deck ด้วย EA; fix deck → improve policy ด้วย RL)

---

## ส่วนที่ 6: Roadmap สำหรับสร้างโมเดล

### ข้อเท็จจริงที่ยืนยันแล้ว vs ที่ยังไม่ยืนยัน

**ยืนยันแล้ว:** 2 categories — Simulation (ไม่มีเงิน, ~16 มิ.ย.–17 ส.ค. 2026) + Strategy (report-based, ถึง ~14 ก.ย. 2026, top 8 เข้า Japan finals; $50k/$30k/$30k-per-top-8 + Google Cloud credits) จัดโดย The Pokémon Company + HEROZ + Matsuo Institute, หนุนโดย Google/Google Cloud/NVIDIA/Kaggle การ์ด ~2,000 ใบ Standard; **5 submissions/วัน; 10 นาที/ผู้เล่น/แมตช์** agent = function รับ observation → return legal action (คุณไม่สร้าง engine คุณสร้างผู้เล่น)

**ยังไม่ยืนยัน (หลัง rules sign-in):** ชื่อ "cabt", ชื่อไฟล์ deck.csv/main.py, การ export replay รายวัน + format JSON, time limit ต่อ turn, **agent รันบน GPU หรือไม่** (มี forum comment เตือนว่าไม่การันตีว่ารันบน GPU) → **วางแผนสำหรับ CPU-only inference**

**Compute strategy:** เทรนหนัก (ใช้ Google Cloud/NVIDIA credits + เครื่อง GPU local สำหรับ self-play RL + BC) แต่ **ship agent เบาบน CPU** (compact policy/value net + bounded ISMCTS ที่เคารพงบเวลาต่อ turn) — เหมือน Pluribus (offline หนัก, 2 CPU live) + Hungry Geese (GPU self-play, CPU+small-MCTS submission)

### 📍 Roadmap (เรียงตามลำดับความสำคัญ)

**Phase 0 — Infrastructure (สัปดาห์ 1–2)**
- wrap SDK เป็น Gym/PettingZoo env
- สร้าง card encoder (per-card feature matrix + action masking)
- ตั้ง fast self-play loop (HandyRL หรือ RLlib)
- สร้าง legal-move-only random/greedy baseline + deterministic rules bot สำหรับ sanity check

**Phase 1 — Imitation bootstrap (สัปดาห์ 2–4)**
- ถ้ามี replay ท็อป → scrape (แบบ Hungry Geese episode scraper) → train supervised policy+value net (cross-entropy บน action, MSE/outcome บน value)
- ถ้าไม่มี replay → generate self-play data จาก ISMCTS/rules bot แล้ว clone
- **Deliverable:** ladder agent ที่ไม่น่าอาย (เกณฑ์: ชนะ rules baseline >70%)

**Phase 2 — Self-play RL (สัปดาห์ 4–8)**
- refine ด้วย PPO หรือ DMC (DouZero-style) / V-trace+UPGO (HandyRL-style)
- เพิ่ม **league training** (main + exploiter) หรือ **R-NaD regularization / KL-anchor ไป frozen BC teacher** กัน strategic cycling
- ใช้ **oracle guiding** (Suphx) — ให้ critic เห็น hidden info เฉพาะตอนเทรน
- reward-shape สำหรับ tempo (prize taken, KO) สู้ sparsity แล้วค่อย anneal ไป pure win/loss

**Phase 3 — Inference-time belief search (สัปดาห์ 6–10)**
- เพิ่ม **ISMCTS** seed ด้วย RL policy prior (PUCT) + belief model over มือ/เด็คคู่ต่อสู้ (card-counting)
- re-determinize ต่อ simulation; pool information-set statistics ลด strategy fusion
- ถ้า endgame เล็ก → ลอง neural-CFR / ReBeL-style resolve บน public belief state
- **cap simulations ตามงบเวลาเข้มงวด** — degrade เป็น raw policy net เมื่อเวลาไม่พอ

**Phase 4 — Deck meta-optimization (ต่อเนื่อง, ตั้งแต่สัปดาห์ 3)**
- maintain candidate deck archive ด้วย **MAP-Elites/EA**
- สร้าง **matchup matrix** จาก self-play + ladder data
- คำนวณ **Nash mixture** → submit deck ที่ exploitability ต่ำ
- re-estimate matrix รายวันจาก replay ใหม่; shift ไป best-response เฉพาะเมื่อ meta นิ่ง
- co-optimize deck+policy ด้วย alternating EA/RL (หรือ PSRO ถ้า compute พอ)

**Phase 5 — Hardening (ต่อเนื่อง)**
- **rule-difference exploitation:** probe simulator หา divergence จากกฎ PTCG จริง + engine quirks (timing, coin-flip RNG, deck-out edge cases)
- watch **σ** ใน TrueSkill — ส่งต่อ (5/วัน) กด σ ลงเมื่อ μ สูง เพราะ leaderboard rank บน μ−kσ

### ✅ คำแนะนำหลัก (Recommendations)

1. **เริ่มเลยด้วย HandyRL + behavioral cloning** บนข้อมูลที่หาได้ — ปีนเร็วสุด + reuse kaggle-environments stack เกณฑ์: BC agent ชนะ rules baseline >70%
2. **ทำ encoder + action masking ให้ถูกก่อนจูน algorithm** — บทเรียน DouZero: representation (card-matrix) สำคัญกว่า RL algorithm ใช้ per-card feature matrix + masked auto-regressive action heads
3. **default ไป robustness แทน exploitation:** ship Nash-mixture deck + equilibrium-regularized (R-NaD/KL-anchored) policy เปลี่ยนไป exploitative เฉพาะเมื่อ daily matchup data แสดง meta ที่นิ่ง/อ่านได้ benchmark: ถ้า worst matchup ของ deck < ~45% win-rate → กลับไป Nash mixture
4. **budget compute แบบ offline-heavy, online-light:** สมมติ CPU-only inference; pre-distill policy เป็น net เล็ก; cap ISMCTS ด้วย wall-clock ถ้ายืนยันว่ามี GPU ตอน inference → scale search ขึ้น
5. **ใช้ exploiter pattern stress-test:** เก็บ "main exploiter" search หารูใน ladder agent ตลอดเวลา แล้ว fold counter กลับด้วย RL — วินัย anti-cycling แบบ AlphaStar แยก #1 ที่นิ่งออกจาก leader ที่เปราะ
6. **สำหรับ Strategy Category report:** document matchup matrix, การหา Nash deck, belief model, anti-cycling argument — ตรงกับ rubric "stability, deck design concept, performance"

---

## ส่วนที่ 7: Repos, Libraries และ Resources

| Repo / Library | ใช้ทำอะไร |
|----------------|-----------|
| **OpenSpiel** (`github.com/google-deepmind/open_spiel`) | CFR/CFR+/MCCFR/Deep CFR, ISMCTS, PSRO, exploitability tools, R-NaD — reference library สำหรับทุกอย่างใน §5.1 |
| **RLCard** (`github.com/datamllab/rlcard`) | card-game RL toolkit (DMC, CFR, DQN) — API ใกล้ PTCG สุด; prototype encoder |
| **DouZero** (`github.com/kwai/DouZero`) | reference DMC + action-encoding + parallel-actor |
| **HandyRL** (`github.com/DeNA/HandyRL`) | distributed self-play RL ด้วย V-trace/UPGO, **มี kaggle-environments support ในตัว** — ทางเร็วสุดไป Kaggle ladder agent |
| **Ray RLlib** | scalable PPO/IMPALA/APPO, league/self-play utilities |
| **PettingZoo / Shimmy** | multi-agent env standard + OpenSpiel bridge |
| **gym-locm** (`github.com/ronaldosvieira/gym-locm`) | self-play drafting+battle RL บน CCG ที่มี deckbuilding รวม — template โครงสร้างดีสุด |

### 📚 Papers สำคัญ (เรียงตามหัวข้อ)

- **CFR:** Zinkevich et al. NIPS 2007; Tammelin 2014 (CFR+); Brown & Sandholm AAAI 2019 (DCFR); Lanctot et al. NIPS 2009 (MCCFR); Brown et al. ICML 2019 (Deep CFR, arXiv:1811.00164)
- **Poker:** Brown & Sandholm Science 2018 (Libratus), Science 2019 (Pluribus); Moravčík et al. Science 2017 (DeepStack); Brown et al. NeurIPS 2020 (ReBeL, arXiv:2007.13544); Schmid et al. Science Advances 2023 (Player of Games)
- **IS-MCTS:** Long et al. AAAI 2010 (PIMC pathologies); Cowling et al. IEEE TCIAIG 2012 (ISMCTS); Ihara et al. IEEE SMC 2018 (ISMCTS on Pokémon, DOI 10.1109/SMC.2018.00375)
- **Stratego:** Perolat et al. Science 2022 (DeepNash/R-NaD, arXiv:2206.15378)
- **Card games:** Zha et al. ICML 2021 (DouZero, arXiv:2106.06135); Li et al. 2020 (Suphx, arXiv:2003.13590); Kowalski & Miernik (LOCM, arXiv:2305.11814); Xi et al. (arXiv:2303.04096)
- **RL/Self-play:** Vinyals et al. Nature 2019 (AlphaStar); AlphaZe* Frontiers in AI 2023

---

## ส่วนที่ 8: ข้อควรระวัง (Caveats)

- **ข้อมูล competition หลายอย่างยังไม่ยืนยัน** (ชื่อ "cabt", ชื่อไฟล์ deck.csv/main.py, การ export replay รายวัน + JSON, time limit ต่อ turn, GPU availability) → อยู่หลัง Kaggle rules wall **ต้องยืนยันจากหน้า Overview/Data/Code + Discord ก่อน commit สถาปัตยกรรม** forum comment สาธารณะชี้ว่า GPU ตอน inference *ไม่*การันตี

- **ผล IS-MCTS-beats-MCTS บน Pokémon (Ihara et al. 2018) เป็นเกมต่อสู้ ไม่ใช่ TCG** → ไม่มี peer-reviewed paper apply ISMCTS กับ Pokémon *Trading* Card Game โดยตรง = transfer โดย analogy

- **Nash guarantee ของ CFR เป็นของ two-player zero-sum** — PTCG เป็น 1v1 zero-sum จึง hold แต่ถ้ามี nuance multiway/non-zero-sum (เช่น draws, time-outs) จะอ่อนลง

- **Self-play ที่ไม่มี anti-cycling safeguard → rock-paper-scissors instability** matchmaking แบบ stochastic ลงโทษ exploiter ที่จูนแคบ → robust (low-exploitability) ปลอดภัยกว่าสำหรับ #1 ที่ยั่งยืน

- **TrueSkill mechanics ที่อธิบายเป็น model มาตรฐาน** — สูตร sort leaderboard จริงของ competition นี้ยังไม่ยืนยัน

- **Compute-cost rules:** มีรายงานว่า competition บังคับ "Reasonableness Standard" ปฏิเสธ submission ที่ใช้ paid training เกินควร → document ค่าเทรนให้ชัดและพอประมาณ

---

*จัดทำเป็นเอกสารอ้างอิงสำหรับการสร้างโมเดล PTCG AI Battle Challenge — รวมข้อมูลทั้งหมดตั้งแต่ต้นการวิเคราะห์*
