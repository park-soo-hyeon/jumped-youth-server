"""
RAG(Retrieval-Augmented Generation) 검색 엔진
====================================================
1. 전체 청년 정책을 OpenAI text-embedding-3-small 모델로 임베딩
2. pickle 파일로 캐싱하여 재시작 시 API 비용 없이 로드
3. 사용자 상태 텍스트를 쿼리 확장(Query Expansion)으로 변환
4. numpy 배치 코사인 유사도로 관련 정책 top-K 반환
5. 무작위 샘플링(기존) 대비 의미적으로 적합한 정책 검색

사용 모델: text-embedding-3-small (비용 효율적, 1536차원)
"""

import json
import os
import pickle

from dotenv import load_dotenv  # .env 파일에서 환경변수 자동 로드
load_dotenv()

import numpy as np
from openai import OpenAI

CACHE_PATH = "policy_embeddings.pkl"
EMBED_MODEL = "text-embedding-3-small"

_cache: dict | None = None  # 모듈 레벨 싱글턴 (프로세스당 1회 로드)


# ── 내부 유틸 ──────────────────────────────────────────────────────────────

def _client() -> OpenAI:
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


def _load_raw_policies() -> list[dict]:
    with open("policies.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("result", {}).get("youthPolicyList", [])


def _policy_to_text(p: dict) -> str:
    """정책 dict → 임베딩용 텍스트 (이름 + 대분류 + 소분류 + 지원내용 앞 300자)"""
    return (
        f"{p.get('plcyNm', '')} "
        f"{p.get('lclsfNm', '')} {p.get('mclsfNm', '')} "
        f"{p.get('plcySprtCn', '')[:300]}"
    ).strip()


# ── 캐시 빌드 / 로드 ───────────────────────────────────────────────────────

def build_cache(force: bool = False) -> dict:
    """
    전체 정책 임베딩을 계산하고 pickle 파일에 저장.
    - force=False: 캐시 파일이 있으면 로드만 수행 (API 비용 절감)
    - force=True : 캐시를 무시하고 재계산
    반환값: {"embeddings": np.ndarray(N, D), "policies": list[dict]}
    """
    global _cache

    if not force and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            _cache = pickle.load(f)
        print(f"[RAG] 캐시 로드 완료: {len(_cache['policies'])}개 정책 임베딩")
        return _cache

    policies = _load_raw_policies()
    texts = [_policy_to_text(p) for p in policies]
    n = len(policies)
    print(f"[RAG] {n}개 정책 임베딩 계산 시작 (최초 실행 시 ~수십 초 소요)...")

    client = _client()
    embeddings: list[list[float]] = []
    batch_size = 100  # OpenAI API 최대 배치

    for i in range(0, n, batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(input=batch, model=EMBED_MODEL)
        embeddings.extend([r.embedding for r in resp.data])
        print(f"[RAG]   {min(i + batch_size, n)}/{n} 완료")

    _cache = {
        "embeddings": np.array(embeddings, dtype=np.float32),  # (N, 1536)
        "policies": policies,
    }
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(_cache, f)
    print(f"[RAG] 임베딩 캐시 저장 완료: {CACHE_PATH}")
    return _cache


def _ensure_cache() -> dict:
    """캐시가 없으면 자동 빌드 (lazy init)"""
    global _cache
    if _cache is None:
        _cache = build_cache()
    return _cache


# ── 핵심 수학 연산 ─────────────────────────────────────────────────────────

def cosine_similarity_batch(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    numpy 벡터 연산으로 배치 코사인 유사도 계산.
    query_vec: (D,)  /  matrix: (N, D)  →  결과: (N,)

    for-loop 대신 행렬 곱(내적)을 사용하여 O(N*D) 연산을 벡터화.
    """
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    normed_matrix = matrix / row_norms          # (N, D) 각 행을 단위 벡터로 정규화
    return normed_matrix @ q_norm               # (N,) 내적 = 코사인 유사도


# ── 쿼리 확장 (Query Expansion) ─────────────────────────────────────────────

def expand_query(mood: str, energy: int, outing: int, comment: str = "") -> str:
    """
    쿼리 확장(Query Expansion): 사용자 입력값을 의미 풍부한 자연어로 변환.

    임베딩 공간에서 recall을 높이는 RAG 전처리 기법.
    단순 숫자 → 설명적 텍스트로 변환하여 정책 설명과의 유사도 향상.
    """
    if energy < 30:
        energy_desc = "에너지가 매우 낮아 집 안에서 할 수 있는"
    elif energy < 70:
        energy_desc = "보통 에너지로 가벼운 외출이 가능한"
    else:
        energy_desc = "충분한 에너지로 적극적 활동이 가능한"

    if outing <= 10:
        outing_desc = "외출이 어려운 상태"
    elif outing <= 30:
        outing_desc = "집 앞까지 외출 가능한 상태"
    elif outing <= 60:
        outing_desc = "동네 산책이 가능한 상태"
    else:
        outing_desc = "대중교통 이용이 가능한 상태"

    query = f"{mood} 기분의 비경제활동 청년, {energy_desc} 활동 지원, {outing_desc}"
    if comment:
        query += f", {comment}"
    return query


# ── 퍼블릭 API ─────────────────────────────────────────────────────────────

def retrieve(
    mood: str,
    energy: int,
    outing: int,
    comment: str = "",
    top_k: int = 10,
) -> list[dict]:
    """
    RAG Retrieval 단계 메인 함수.

    Pipeline:
      [사용자 상태] → [쿼리 확장] → [임베딩] → [코사인 유사도 검색] → [top_k 정책]

    무작위 샘플링 대비:
      - 사용자 상태와 의미적으로 관련 높은 정책만 선별
      - 에너지/외출 수준에 맞는 정책 우선 순위화
      - 검색 결과에 유사도 스코어 포함 (GPT 프롬프트 투명성)
    """
    cache = _ensure_cache()
    query_text = expand_query(mood, energy, outing, comment)

    # 쿼리 임베딩
    resp = _client().embeddings.create(input=[query_text], model=EMBED_MODEL)
    query_vec = np.array(resp.data[0].embedding, dtype=np.float32)

    # 코사인 유사도 계산 및 정렬
    sims = cosine_similarity_batch(query_vec, cache["embeddings"])
    top_indices = np.argsort(sims)[::-1][:top_k]

    results = []
    for idx in top_indices:
        p = cache["policies"][idx]
        results.append({
            "이름": p.get("plcyNm", ""),
            "분류": f"{p.get('lclsfNm', '')} > {p.get('mclsfNm', '')}",
            "지원내용": p.get("plcySprtCn", "")[:200],
            "신청방법": p.get("plcyAplyMthdCn", "")[:100],
            "링크": p.get("aplyUrlAddr", "") or p.get("refUrlAddr1", ""),
            "대상연령": f"{p.get('sprtTrgtMinAge', '?')}~{p.get('sprtTrgtMaxAge', '?')}세",
            "유사도": round(float(sims[idx]), 4),  # 투명성: 검색 품질 확인용
        })
    return results
