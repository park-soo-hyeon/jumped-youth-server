"""
뛰었음 청년 - Flask API 서버
============================
RAG(Retrieval-Augmented Generation) 파이프라인:
  사용자 상태 → 쿼리 확장 → 임베딩 검색 → GPT-4o-mini 생성

주요 Python AI 기술:
  - OpenAI Embeddings (text-embedding-3-small)
  - numpy 코사인 유사도 검색
  - scikit-learn K-Means 행동 패턴 클러스터링
  - GPT-4o-mini structured JSON 출력
  - 회복 단계 분류 (초기/중간/확장)
"""

import json
import os
import sqlite3
from collections import Counter
from datetime import datetime, timedelta

from dotenv import load_dotenv  # .env 파일에서 환경변수 자동 로드
load_dotenv()  # back/.env 파일을 읽어서 os.environ에 등록

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import rag_engine  # RAG 검색 엔진 (rag_engine.py)

# 프로덕션 빌드 시 React 정적 파일 경로 (../front/build)
STATIC_DIR = os.path.join(os.path.dirname(__file__), '..', 'front', 'build')

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='')
CORS(app)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


# ══════════════════════════════════════════════════════════════════════════════
# DB
# ══════════════════════════════════════════════════════════════════════════════

def get_db():
    conn = sqlite3.connect("history.db")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  date TEXT, mood TEXT, energy INTEGER, outing INTEGER,
                  comment TEXT, mission TEXT, mission_detail TEXT,
                  energy_tag TEXT, policy_name TEXT, policy_link TEXT,
                  is_completed INTEGER DEFAULT 0)""")
    existing = {row[1] for row in c.execute("PRAGMA table_info(history)")}
    for col, dfn in [
        ("policy_name",    "TEXT"),
        ("policy_link",    "TEXT"),
        ("comment",        "TEXT"),
        ("mission_detail", "TEXT"),
        ("energy_tag",     "TEXT"),
    ]:
        if col not in existing:
            c.execute(f"ALTER TABLE history ADD COLUMN {col} {dfn}")
    conn.commit()
    conn.close()


init_db()
rag_engine.build_cache()  # gunicorn 등 WSGI 서버로 실행 시에도 캐시 로드


# ══════════════════════════════════════════════════════════════════════════════
# 회복 단계 분류 (PPT 사용자 시나리오 기반)
# ══════════════════════════════════════════════════════════════════════════════

def get_recovery_stage(conn) -> tuple[str, str]:
    """
    완료된 미션 수를 기준으로 회복 단계를 분류한다.
    PPT 슬라이드 10의 사용자 시나리오(초기→중간→확장)를 규칙 기반으로 구현.

    Returns: (stage_name, description)
    """
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM history WHERE is_completed = 1")
    n = c.fetchone()[0]

    if n < 3:
        return "초기", "작은 행동을 반복하며 일상 습관을 만드는 단계예요"
    elif n < 10:
        return "중간", "정책·훈련 정보를 탐색하기 시작할 수 있는 단계예요"
    else:
        return "확장", "고용24 기반 진로·채용 연계를 시도해볼 수 있는 단계예요"


# ══════════════════════════════════════════════════════════════════════════════
# API 엔드포인트
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/recommend", methods=["POST"])
def recommend():
    """
    AI 미션 추천 (RAG 파이프라인 적용).

    기존 무작위 샘플링 → RAG 의미 검색으로 대체:
      1. rag_engine.retrieve()로 사용자 상태와 의미적으로 유사한 정책 top-10 검색
      2. 회복 단계 반영하여 GPT 시스템 프롬프트 개인화
      3. GPT-4o-mini로 2가지 미션 생성 (JSON mode)

    preview=True: DB 저장 없이 AI 결과만 반환 (사용자가 확정 전 미리보기)
    preview=False: 첫 번째 미션을 DB에 저장
    """
    data = request.json
    user_mood = data.get("mood")
    energy    = int(data.get("energy", 50))
    outing    = int(data.get("outing", 30))
    comment   = data.get("comment", "")
    preview   = data.get("preview", False)

    try:
        # ── RAG Retrieval ──────────────────────────────────────────────────
        # 무작위 샘플링 대신 쿼리 확장 + 임베딩 코사인 유사도 검색
        rag_policies = rag_engine.retrieve(user_mood, energy, outing, comment, top_k=10)
        policies_text = json.dumps(rag_policies, ensure_ascii=False, indent=2)

        # ── 회복 단계 개인화 ───────────────────────────────────────────────
        conn = get_db()
        stage, stage_desc = get_recovery_stage(conn)
        conn.close()

        # ── GPT-4o-mini Augmented Generation ──────────────────────────────
        system_prompt = f"""너는 은둔 청년을 돕는 다정한 AI 멘토야.
        사용자는 현재 '{stage}' 단계({stage_desc})에 있어.
        아래 정책 목록은 사용자 상태와 RAG(임베딩 유사도 검색)로 추출된 관련 정책이야.
        이 중 가장 적합한 것들을 골라 서로 다른 2가지 미션을 추천해줘.
        미션은 사용자의 에너지·외출 수준에 맞게 아주 작고 부담 없어야 해.
        energyTag는 반드시 "낮은 난이도", "보통 난이도", "높은 난이도" 셋 중 하나만 사용해.
        반드시 아래 JSON 형식으로만 응답해:
        {{
        "aiMessage": "따뜻하고 공감하는 위로 메시지 1-2문장",
        "missions": [
            {{"mission": "첫 번째 미션 제목", "missionDetail": "따뜻하고 부담 없는 설명 2-3문장", "energyTag": "낮은 난이도/보통 난이도/높은 난이도 중 하나", "policyName": "관련 정책명", "policyLink": "링크URL"}},
            {{"mission": "두 번째 미션 제목", "missionDetail": "따뜻하고 부담 없는 설명 2-3문장", "energyTag": "낮은 난이도/보통 난이도/높은 난이도 중 하나", "policyName": "관련 정책명", "policyLink": "링크URL"}}
        ]
        }}"""

        user_prompt = f"""사용자 상태:
        - 기분: {user_mood}
        - 에너지 레벨: {energy}% (0=매우 낮음, 100=매우 높음)
        - 외출 가능 지수: {outing}% (0=외출 불가, 100=자유로운 외출)
        - 회복 단계: {stage}
        {f'- 오늘 한마디: {comment}' if comment else ''}

RAG 검색된 관련 정책 목록 (유사도 순):
{policies_text}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )

        ai_reply = json.loads(response.choices[0].message.content)

        # 구버전 단일 미션 응답 호환 처리
        if "missions" not in ai_reply and "mission" in ai_reply:
            ai_reply = {
                "aiMessage": ai_reply.get("aiMessage", ""),
                "missions": [{
                    "mission":       ai_reply.get("mission", ""),
                    "missionDetail": ai_reply.get("missionDetail", ""),
                    "energyTag":     ai_reply.get("energyTag", ""),
                    "policyName":    ai_reply.get("policyName", ""),
                    "policyLink":    ai_reply.get("policyLink", ""),
                }],
            }

        if preview:
            return jsonify({"data": ai_reply})

        # preview=False: 첫 번째 미션 DB 저장
        first = (ai_reply.get("missions") or [{}])[0]
        conn = get_db()
        c = conn.cursor()
        c.execute(
            "INSERT INTO history (date, mood, energy, outing, comment, mission, mission_detail, energy_tag, policy_name, policy_link) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                datetime.now().strftime("%Y-%m-%d"),
                user_mood, energy, outing, comment,
                first.get("mission", ""),
                first.get("missionDetail", ""),
                first.get("energyTag", ""),
                first.get("policyName", ""),
                first.get("policyLink", ""),
            ),
        )
        row_id = c.lastrowid
        conn.commit()
        conn.close()
        return jsonify({"id": row_id, "data": ai_reply})

    except Exception as e:
        import traceback
        traceback.print_exc()  # 백엔드 터미널에 전체 스택 출력
        return jsonify({"error": str(e)}), 500


@app.route("/api/missions", methods=["POST"])
def save_mission():
    """사용자가 미리보기에서 선택한 미션을 DB에 확정 저장"""
    data = request.json
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "INSERT INTO history (date, mood, energy, outing, comment, mission, mission_detail, energy_tag, policy_name, policy_link) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            datetime.now().strftime("%Y-%m-%d"),
            data.get("mood", ""), int(data.get("energy", 50)), int(data.get("outing", 30)),
            data.get("comment", ""),       data.get("mission", ""),
            data.get("missionDetail", ""), data.get("energyTag", ""),
            data.get("policyName", ""),    data.get("policyLink", ""),
        ),
    )
    row_id = c.lastrowid
    conn.commit()
    conn.close()
    return jsonify({"id": row_id})


@app.route("/api/complete", methods=["POST"])
def complete_mission():
    """미션 완료 처리"""
    mission_id = request.json.get("id")
    conn = get_db()
    conn.execute("UPDATE history SET is_completed = 1 WHERE id = ?", (mission_id,))
    conn.commit()
    conn.close()
    return jsonify({"message": "미션 완료!"})


@app.route("/api/history", methods=["GET"])
def get_history():
    """최근 30개 히스토리 반환"""
    conn = get_db()
    rows = conn.execute(
        "SELECT id, date, mood, energy, outing, comment, mission, mission_detail, "
        "energy_tag, policy_name, policy_link, is_completed "
        "FROM history ORDER BY id DESC LIMIT 30"
    ).fetchall()
    conn.close()
    return jsonify([
        {
            "id": r[0], "date": r[1], "mood": r[2],
            "energy": r[3], "outing": r[4], "comment": r[5],
            "mission": r[6], "missionDetail": r[7], "energyTag": r[8],
            "policyName": r[9], "policyLink": r[10],
            "isCompleted": bool(r[11]),
        }
        for r in rows
    ])


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """streak(연속일수) + 주간 요일별 통계 + 최근 완료 미션"""
    conn = get_db()
    c = conn.cursor()
    today = datetime.now()

    # 이번 주 (월~일) 날짜별 통계
    week_start = today - timedelta(days=today.weekday())
    weekly = {}
    for i in range(7):
        d = (week_start + timedelta(days=i)).strftime("%Y-%m-%d")
        row = c.execute(
            "SELECT COUNT(*), COALESCE(SUM(is_completed),0) FROM history WHERE date=?", (d,)
        ).fetchone()
        weekly[d] = {"total": row[0] or 0, "completed": int(row[1] or 0)}

    # 연속 완료 일수 (오늘부터 역산)
    streak, check = 0, today
    while True:
        d = check.strftime("%Y-%m-%d")
        cnt = c.execute(
            "SELECT COUNT(*) FROM history WHERE date=? AND is_completed=1", (d,)
        ).fetchone()[0]
        if cnt > 0:
            streak += 1
            check -= timedelta(days=1)
        else:
            break

    # 가장 최근 완료 미션
    row = c.execute(
        "SELECT mission FROM history WHERE is_completed=1 ORDER BY id DESC LIMIT 1"
    ).fetchone()

    conn.close()
    return jsonify({"streak": streak, "weekly": weekly, "lastMission": row[0] if row else None})


@app.route("/api/analysis", methods=["GET"])
def get_analysis():
    """
    사용자 행동 패턴 분석 - Python AI/ML 데이터 분석 레이어.

    사용 기술:
      - numpy: 완료율·평균 에너지/외출 통계 계산 (벡터 연산)
      - scikit-learn KMeans: 에너지×외출 2D 공간에서 행동 패턴 클러스터링
      - scikit-learn StandardScaler: 피처 스케일 정규화
      - collections.Counter: 기분별 빈도 집계
    """
    conn = get_db()
    c = conn.cursor()

    rows = c.execute(
        "SELECT mood, energy, outing, is_completed FROM history"
    ).fetchall()

    stage, stage_desc = get_recovery_stage(conn)
    conn.close()

    if not rows:
        return jsonify({
            "stage": stage, "stageDescription": stage_desc,
            "totalMissions": 0, "totalCompleted": 0,
            "completionRate": 0, "moodStats": {},
            "avgEnergyOnSuccess": 0, "avgOutingOnSuccess": 0,
            "clusterInsights": None,
        })

    # ── numpy 통계 분석 ────────────────────────────────────────────────────
    moods     = [r[0] for r in rows]
    energies  = np.array([r[1] for r in rows], dtype=float)
    outings   = np.array([r[2] for r in rows], dtype=float)
    completed = np.array([r[3] for r in rows], dtype=float)

    total           = len(rows)
    total_completed = int(np.sum(completed))
    completion_rate = float(np.mean(completed))

    # 기분별 완료율 (numpy boolean indexing)
    mood_stats = {}
    for mood in set(moods):
        mask = np.array([m == mood for m in moods])
        mood_stats[mood] = {
            "total":          int(np.sum(mask)),
            "completed":      int(np.sum(completed[mask])),
            "completionRate": round(float(np.mean(completed[mask])), 2),
        }

    # 성공(완료) 미션에서의 평균 에너지·외출 수준
    success_mask = completed == 1
    avg_energy_on_success = round(float(np.mean(energies[success_mask])), 1) if np.any(success_mask) else 0
    avg_outing_on_success = round(float(np.mean(outings[success_mask])), 1)  if np.any(success_mask) else 0

    # 가장 완료율 높은 기분
    best_mood = max(mood_stats, key=lambda m: mood_stats[m]["completionRate"]) if mood_stats else None

    # ── scikit-learn K-Means 클러스터링 ───────────────────────────────────
    # 에너지(x) × 외출(y) 2D 공간에서 행동 패턴을 자동 분류
    cluster_insights = None
    if total >= 5:
        X = np.column_stack([energies, outings])          # (N, 2)
        scaler  = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_clusters = min(3, total)
        kmeans  = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels  = kmeans.fit_predict(X_scaled)

        # 클러스터 중심을 원래 스케일로 역변환
        centers_original = scaler.inverse_transform(kmeans.cluster_centers_)

        # 클러스터별 사람이 읽기 쉬운 이름 부여
        cluster_names = []
        for ce, co in centers_original:
            if ce < 35:
                cluster_names.append("낮은 에너지 패턴")
            elif ce < 65:
                cluster_names.append("중간 에너지 패턴")
            else:
                cluster_names.append("활동적 패턴")

        # 클러스터별 완료율
        cluster_rates = []
        for k in range(n_clusters):
            mask_k = labels == k
            rate = float(np.mean(completed[mask_k])) if np.any(mask_k) else 0
            cluster_rates.append(round(rate, 2))

        cluster_insights = {
            "labels":        labels.tolist(),
            "centers":       [[round(e, 1), round(o, 1)] for e, o in centers_original],
            "names":         cluster_names,
            "completionRates": cluster_rates,
            "description":   f"{n_clusters}개 행동 패턴 그룹으로 분류 (에너지×외출 K-Means)",
        }

    return jsonify({
        "stage":               stage,
        "stageDescription":    stage_desc,
        "totalMissions":       total,
        "totalCompleted":      total_completed,
        "completionRate":      round(completion_rate, 2),
        "moodStats":           mood_stats,
        "bestMood":            best_mood,
        "avgEnergyOnSuccess":  avg_energy_on_success,
        "avgOutingOnSuccess":  avg_outing_on_success,
        "clusterInsights":     cluster_insights,
    })


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    """프로덕션 빌드 파일이 있을 경우 React SPA 서빙 (front/build 디렉토리)"""
    if path and os.path.exists(os.path.join(STATIC_DIR, path)):
        return send_from_directory(STATIC_DIR, path)
    if os.path.exists(os.path.join(STATIC_DIR, 'index.html')):
        return send_from_directory(STATIC_DIR, 'index.html')
    return jsonify({"message": "뛰었음 청년 API 서버"}), 200


if __name__ == "__main__":
    # 서버 시작 시 RAG 임베딩 캐시 미리 로드 (없으면 빌드)
    print("[서버] RAG 임베딩 캐시 준비 중...")
    rag_engine.build_cache()
    print("[서버] 준비 완료. http://localhost:8000 에서 실행 중")
    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=False)
