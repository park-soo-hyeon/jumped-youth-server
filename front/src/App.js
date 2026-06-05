/**
 * 뛰었음 청년 - 프론트엔드 메인 컴포넌트
 * ==========================================
 * 서비스 개요: RAG/LLM 기반 비경제활동 청년(쉬었음 청년) 행동 유도 웹앱
 * 핵심 기능:
 *   1. 오늘의 상태 입력 (기분·에너지·외출 가능 정도)
 *   2. AI 맞춤형 회복 미션 추천 (Backend RAG → GPT-4o-mini)
 *   3. 미션 수락/완료 흐름 + 히스토리 기록
 *   4. 회복 단계 분류 및 K-Means 행동 패턴 분석 결과 시각화
 *
 * UI 구조: 3단 컬럼 레이아웃 (PPT 슬라이드 8 기반)
 *   좌측: 상태 입력 폼 + 통계 카드
 *   중앙: AI 미션 카드 (2개 동시 표시)
 *   우측: 주간 캘린더 + K-Means 분석 + 히스토리
 *
 * 기술 스택: React (CRA), axios, CSS-in-JS (inline style)
 */

import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

/**
 * 백엔드 API 주소.
 * - 로컬 개발: http://127.0.0.1:8000 (Flask 개발 서버)
 * - Railway 배포: '' (빈 문자열) → Flask가 React 빌드도 함께 서빙하므로 상대 경로 사용
 * - 환경변수 REACT_APP_API_BASE로 외부에서 덮어쓰기 가능
 */
const API_BASE = process.env.REACT_APP_API_BASE
  ?? (window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost'
    ? 'http://127.0.0.1:8000'  // 로컬 개발환경
    : '');                      // 배포 환경: 같은 도메인(Flask가 React + API 동시 서빙)

// ══════════════════════════════════════════════════════════════════════════════
// 상수 정의
// ══════════════════════════════════════════════════════════════════════════════

/**
 * 기분 선택 버튼 3종 (PPT 기획)
 * - 각 기분에 맞는 배경색·테두리색·활성화 색상을 정의
 * - 색상은 기분의 심리적 연상(파랑=우울, 노랑=보통, 초록=편안)을 반영
 */
const MOOD_OPTIONS = [
  { value: '우울해', emoji: '😞', bg: '#E8F0FE', activeBg: '#B5C8F0', border: '#6C9FE8' },
  { value: '보통',   emoji: '😐', bg: '#FFF8E8', activeBg: '#F5DDA0', border: '#E8B44A' },
  { value: '편안함', emoji: '😊', bg: '#EAFAF1', activeBg: '#A8DDB5', border: '#4CAF50' },
];

/**
 * 외출 가능 정도 드롭다운 옵션 4단계 (PPT 기획)
 * - value는 백엔드로 전달되는 수치(0~100)로, GPT 프롬프트에 사용됨
 * - 단계를 숫자로 변환하여 쿼리 확장(Query Expansion)에 활용
 */
const OUTING_OPTIONS = [
  { label: '집 밖으로 못 나가겠어요',     value: 10 },
  { label: '집 앞까지 나갈 수 있어요',     value: 30 },
  { label: '동네 산책이 가능해요',         value: 60 },
  { label: '대중교통도 이용할 수 있어요',   value: 90 },
];

/** 주간 캘린더 요일 표시용 (월요일 시작 기준) */
const DAYS_KO = ['월', '화', '수', '목', '금', '토', '일'];

/**
 * 미션 난이도(energyTag) 별 표시 스타일
 * - GPT가 생성하는 energyTag 값과 1:1 매핑
 * - 초록=쉬움, 노랑=보통, 빨강=어려움 (직관적 색상 코딩)
 */
const ENERGY_TAG_STYLES = {
  '낮은 난이도': { bg: '#D4EDDA', color: '#155724' },
  '보통 난이도': { bg: '#FFF3CD', color: '#856404' },
  '높은 난이도': { bg: '#FFE0E0', color: '#C0392B' },
};

/**
 * 회복 단계 배지 스타일 (PPT 사용자 시나리오 슬라이드 기반)
 * - 초기: 작은 행동 반복 → 씨앗(🌱) 이미지, 노란 배경
 * - 중간: 정책·훈련 정보 탐색 → 걷기(🚶) 이미지, 초록 배경
 * - 확장: 고용24 진로·채용 연계 → 로켓(🚀) 이미지, 파란 배경
 */
const STAGE_STYLES = {
  '초기': { bg: '#FFF3CD', color: '#856404', icon: '🌱' },
  '중간': { bg: '#D4EDDA', color: '#155724', icon: '🚶' },
  '확장': { bg: '#CCE5FF', color: '#004085', icon: '🚀' },
};

// ══════════════════════════════════════════════════════════════════════════════
// 유틸리티 함수
// ══════════════════════════════════════════════════════════════════════════════

/**
 * 사용자 입력값(기분·에너지·외출)을 기반으로 헤더 상태 태그를 자동 생성.
 * - 헤더 우측에 chip 형태로 표시되어 현재 상태를 한눈에 보여줌
 * - 에너지/외출 임계값은 사용자 경험상 의미 있는 구간으로 구분
 *
 * @param {string} mood - 기분 ('우울해' | '보통' | '편안함')
 * @param {number} energy - 에너지 수준 (0~100)
 * @param {number} outingValue - 외출 가능 수치 (10|30|60|90)
 * @returns {Array<{label, bg, color}>} 태그 배열
 */
function computeStateTags(mood, energy, outingValue) {
  const tags = [];

  // 기분 태그 (3가지 고정 선택)
  if (mood === '우울해')    tags.push({ label: '기분 저조',    bg: '#FFE0E0', color: '#C0392B' });
  else if (mood === '보통') tags.push({ label: '안정적인 기분', bg: '#FFF3CD', color: '#856404' });
  else if (mood === '편안함') tags.push({ label: '편안한 상태', bg: '#D4EDDA', color: '#155724' });

  // 에너지 태그 (30/70 기준 3구간)
  if (energy < 30)      tags.push({ label: '낮은 에너지', bg: '#FFE0E0', color: '#C0392B' });
  else if (energy < 70) tags.push({ label: '보통 에너지', bg: '#FFF3CD', color: '#856404' });
  else                  tags.push({ label: '높은 에너지', bg: '#D4EDDA', color: '#155724' });

  // 외출 가능 태그 (4단계 드롭다운 값 기준)
  if (outingValue <= 10)      tags.push({ label: '외출 어려운 상태', bg: '#FFE0E0', color: '#C0392B' });
  else if (outingValue <= 30) tags.push({ label: '가벼운 외출 가능', bg: '#FFF3CD', color: '#856404' });
  else if (outingValue <= 60) tags.push({ label: '동네 산책 가능',   bg: '#D4EDDA', color: '#155724' });
  else                        tags.push({ label: '자유로운 외출',    bg: '#D4EDDA', color: '#155724' });

  return tags;
}

// ══════════════════════════════════════════════════════════════════════════════
// 서브 컴포넌트
// ══════════════════════════════════════════════════════════════════════════════

/**
 * 주간 캘린더 컴포넌트
 * - 이번 주 월~일의 미션 완료 현황을 원형 아이콘으로 시각화
 * - 완료(보라 ✓) / 진행중(빨강 ·) / 빈 날(회색) / 오늘(테두리 강조)
 *
 * @param {{weekly: Object}} props
 *   weekly: {날짜문자열: {total: number, completed: number}} 형태의 주간 데이터
 */
function WeeklyCalendar({ weekly }) {
  if (!weekly) return null;

  const today = new Date().toISOString().slice(0, 10); // 오늘 날짜 (YYYY-MM-DD)
  // 날짜 오름차순 정렬 (월→일 순서 보장)
  const entries = Object.entries(weekly).sort(([a], [b]) => a.localeCompare(b));

  return (
    <div>
      <p style={{ fontSize: '12px', color: '#999', margin: '0 0 8px', fontWeight: 'bold' }}>이번 주 미션 현황</p>
      <div style={{ display: 'flex', gap: '4px', justifyContent: 'space-between', marginBottom: '16px' }}>
        {entries.map(([date, stat], i) => {
          const isToday   = date === today;
          const done      = stat.completed > 0; // 완료된 미션이 있으면 초록 체크
          const hasMission = stat.total > 0;    // 미션이 있지만 미완료면 빨간 점

          return (
            <div key={date} style={{ flex: 1, textAlign: 'center' }}>
              {/* 요일 레이블 (오늘이면 보라색 강조) */}
              <div style={{
                fontSize: '11px',
                color: isToday ? '#6C5CE7' : '#aaa',
                marginBottom: '4px',
                fontWeight: isToday ? 'bold' : 'normal',
              }}>
                {DAYS_KO[i]}
              </div>
              {/* 상태 원형 아이콘 */}
              <div style={{
                width: '28px', height: '28px', borderRadius: '50%', margin: '0 auto',
                backgroundColor: done ? '#6C5CE7' : hasMission ? '#FFB3BA' : isToday ? '#EDE7FF' : '#F0F0F0',
                border: isToday ? '2px solid #6C5CE7' : '2px solid transparent',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '13px', color: done ? 'white' : hasMission ? '#C0392B' : '#ccc',
              }}>
                {done ? '✓' : hasMission ? '·' : ''}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/**
 * 미션 카드 컴포넌트
 * - preview 모드: 2개 카드 동시 표시, 각 카드에 "이 미션 할래요" / "다른 미션 보기" 버튼
 * - active 모드: 확정된 미션 1개, "미션 완료하기!" 버튼
 * - isActive=false → onAccept(카드별 수락) + onRefresh(새 추천)
 * - isActive=true  → onComplete(완료 처리)
 *
 * @param {Object} props
 *   mission    - {mission, missionDetail, energyTag, policyName, policyLink}
 *   isActive   - 현재 진행 중인 확정 미션 여부
 *   onAccept   - "이 미션 할래요" 클릭 핸들러 (preview 전용)
 *   onRefresh  - "다른 미션 보기" 클릭 핸들러 (preview 전용)
 *   onComplete - "미션 완료하기" 클릭 핸들러 (active 전용)
 *   loading    - AI 응답 대기 중 여부 (버튼 비활성화)
 */
function MissionCard({ mission, isActive, onAccept, onRefresh, onComplete, loading }) {
  // energyTag에 해당하는 색상 스타일 조회 (없으면 기본 회색)
  const tagStyle = ENERGY_TAG_STYLES[mission.energyTag] || { bg: '#F0F0F0', color: '#666' };

  return (
    <div style={{
      border: `2px solid ${isActive ? '#6C5CE7' : '#E8E0FF'}`, // 진행 중이면 진한 테두리
      borderRadius: '16px', padding: '20px', backgroundColor: 'white', marginBottom: '14px',
    }}>
      {/* ── 뱃지 영역 ── */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '12px', flexWrap: 'wrap' }}>
        {/* 항상 표시되는 "오늘의 추천" 보라색 뱃지 */}
        <span style={{
          padding: '4px 12px', backgroundColor: '#6C5CE7', color: 'white',
          borderRadius: '20px', fontSize: '11px', fontWeight: '700',
        }}>
          오늘의 추천
        </span>

        {/* GPT가 생성한 난이도 태그 (낮은/보통/높은 난이도) */}
        {mission.energyTag && (
          <span style={{
            padding: '4px 12px', backgroundColor: tagStyle.bg, color: tagStyle.color,
            borderRadius: '20px', fontSize: '11px', fontWeight: '600',
          }}>
            {mission.energyTag}
          </span>
        )}

        {/* 확정된 미션일 때만 "진행 중" 초록 뱃지 표시 */}
        {isActive && (
          <span style={{
            padding: '4px 12px', backgroundColor: '#EAFAF1', color: '#1A7740',
            borderRadius: '20px', fontSize: '11px', fontWeight: '600',
          }}>
            진행 중
          </span>
        )}
      </div>

      {/* ── 미션 제목 ── */}
      <h2 style={{ margin: '0 0 10px', fontSize: '19px', color: '#2C2C2C', fontWeight: '800', lineHeight: '1.35' }}>
        {mission.mission}
      </h2>

      {/* ── 미션 상세 설명 (GPT 생성, 2-3문장) ── */}
      {mission.missionDetail && (
        <p style={{ margin: '0 0 14px', fontSize: '13px', color: '#666', lineHeight: '1.75' }}>
          {mission.missionDetail}
        </p>
      )}

      {/* ── 관련 청년 정책 정보 (RAG로 검색된 정책) ── */}
      {mission.policyName && (
        <div style={{ backgroundColor: '#F8F9FA', padding: '11px 13px', borderRadius: '10px', marginBottom: '14px' }}>
          <p style={{ margin: '0 0 3px', fontSize: '10px', color: '#999', fontWeight: '600' }}>관련 지원 정책</p>
          <p style={{ margin: '0 0 4px', fontSize: '13px', fontWeight: '700', color: '#444' }}>
            📋 {mission.policyName}
          </p>
          {/* 정책 신청 링크 (RAG 검색 결과에서 추출된 URL) */}
          {mission.policyLink ? (
            <a href={mission.policyLink} target="_blank" rel="noreferrer"
              style={{ fontSize: '12px', color: '#6C5CE7', textDecoration: 'none', fontWeight: '600' }}>
              신청하러 가기 →
            </a>
          ) : (
            <span style={{ fontSize: '12px', color: '#ccc' }}>링크 없음</span>
          )}
        </div>
      )}

      {/* ── 버튼: preview 상태 (미리보기, DB 미저장) ── */}
      {!isActive && onAccept && (
        <div style={{ display: 'flex', gap: '10px' }}>
          {/* 이 미션 수락 → POST /api/missions → active 상태로 전환 */}
          <button
            onClick={onAccept}
            style={{
              flex: 1, padding: '12px', borderRadius: '12px', border: 'none',
              backgroundColor: '#6C5CE7', color: 'white', fontSize: '13px',
              fontWeight: '700', cursor: 'pointer',
            }}
          >
            이 미션 할래요 ✔️
          </button>
          {/* 새 미션 추천 요청 → POST /api/recommend (preview=true) */}
          <button
            onClick={onRefresh}
            disabled={loading}
            style={{
              flex: 1, padding: '12px', borderRadius: '12px',
              border: '2px solid #6C5CE7', backgroundColor: 'white',
              color: '#6C5CE7', fontSize: '13px', fontWeight: '700',
              cursor: loading ? 'not-allowed' : 'pointer',
            }}
          >
            다른 미션 보기
          </button>
        </div>
      )}

      {/* ── 버튼: active 상태 (확정 미션, DB 저장됨) ── */}
      {isActive && onComplete && (
        /* 완료 클릭 → POST /api/complete → is_completed=1 업데이트 */
        <button
          onClick={onComplete}
          style={{
            width: '100%', padding: '14px', borderRadius: '12px', border: 'none',
            backgroundColor: '#FF7E79', color: 'white', fontSize: '15px',
            fontWeight: '700', cursor: 'pointer',
          }}
        >
          미션 완료하기! 🎉
        </button>
      )}
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// 메인 앱 컴포넌트
// ══════════════════════════════════════════════════════════════════════════════

/**
 * App - 뛰었음 청년 메인 페이지
 *
 * 상태(State) 구조:
 *   - mood / energy / outingValue / comment: 사용자 입력 상태
 *   - preview: AI 추천 결과 (미리보기, DB 미저장) {aiMessage, missions: [...]}
 *   - active:  수락 확정된 미션 (DB 저장, id 포함)
 *   - history: GET /api/history 결과 (최근 30개)
 *   - stats:   GET /api/stats 결과 (streak, weekly, lastMission)
 *   - analysis: GET /api/analysis 결과 (회복단계, 완료율, K-Means 클러스터)
 */
export default function App() {
  // ── 사용자 입력 상태 ───────────────────────────────────────────────────────
  const [mood, setMood] = useState('');          // 기분 선택 (3가지 버튼)
  const [energy, setEnergy] = useState(50);      // 에너지 수준 (0~100 슬라이더)
  const [outingValue, setOutingValue] = useState(30); // 외출 가능 수치 (드롭다운)
  const [comment, setComment] = useState('');    // 오늘의 한마디 (선택 입력)

  // ── 미션 상태 (preview → active 단방향 흐름) ──────────────────────────────
  // preview: RAG+GPT 결과를 미리 보여주는 상태. DB 저장 전. 사용자가 수락하면 active로 전환.
  // active:  /api/missions로 DB 저장 완료된 미션. id를 가지고 있어 완료 처리 가능.
  const [preview, setPreview] = useState(null);
  const [active, setActive] = useState(null);
  const [loading, setLoading] = useState(false); // AI 응답 대기 상태

  // ── 데이터 상태 ───────────────────────────────────────────────────────────
  const [history, setHistory] = useState([]);   // 최근 30개 미션 이력
  const [stats, setStats] = useState(null);     // streak, weekly 캘린더, 최근 완료 미션
  // analysis: 백엔드 numpy/sklearn 분석 결과 (회복단계, 완료율, K-Means 클러스터)
  const [analysis, setAnalysis] = useState(null);

  // ── 데이터 로딩 ───────────────────────────────────────────────────────────

  /**
   * 히스토리·통계·분석 데이터를 병렬로 가져오는 함수.
   * useCallback으로 메모이제이션하여 useEffect 의존성 배열에 안전하게 사용.
   * Promise.all로 3개 API를 동시에 호출하여 응답 속도 최적화.
   */
  const fetchData = useCallback(async () => {
    try {
      const [histRes, statsRes, analysisRes] = await Promise.all([
        axios.get(`${API_BASE}/api/history`),  // 최근 미션 이력
        axios.get(`${API_BASE}/api/stats`),    // streak + 주간 캘린더
        axios.get(`${API_BASE}/api/analysis`), // numpy/sklearn 패턴 분석
      ]);
      setHistory(histRes.data);
      setStats(statsRes.data);
      setAnalysis(analysisRes.data);
    } catch (e) {
      console.error('데이터 로딩 실패:', e);
    }
  }, []);

  // 첫 렌더링 시 데이터 로드
  useEffect(() => { fetchData(); }, [fetchData]);

  // ── 미션 추천 (RAG 파이프라인 호출) ──────────────────────────────────────

  /**
   * AI 미션 추천 요청.
   * preview=true로 전송하여 DB 저장 없이 AI 결과만 수신.
   * 백엔드에서:
   *   1. rag_engine.retrieve()로 관련 정책 top-10 검색
   *   2. 회복 단계 정보와 함께 GPT-4o-mini에 전달
   *   3. {aiMessage, missions: [{...}, {...}]} 반환
   */
  const fetchMission = async () => {
    if (!mood) { alert('기분을 선택해주세요!'); return; }
    setLoading(true);
    setPreview(null); // 이전 미리보기 초기화
    setActive(null);  // 진행 중 미션도 초기화 (새 추천 시작)
    try {
      const res = await axios.post(`${API_BASE}/api/recommend`, {
        mood, energy, outing: outingValue, comment,
        preview: true,
      });
      setPreview(res.data.data);
    } catch (e) {
      // 실제 오류 원인을 순서대로 추출하여 표시
      const serverMsg = e.response?.data?.error;   // Flask 500 에러 메시지
      const netMsg   = e.code === 'ERR_NETWORK'    // 백엔드 연결 실패
        ? '백엔드 서버에 연결할 수 없습니다. python app.py가 실행 중인지 확인하세요.'
        : null;
      const detail = serverMsg || netMsg || e.message || '알 수 없는 오류';
      alert(`오류 발생:\n${detail}`);
      console.error('[추천 오류]', e.response?.data || e.message);
    } finally {
      setLoading(false);
    }
  };

  // ── 미션 수락 (preview → active 전환) ────────────────────────────────────

  /**
   * 사용자가 선택한 미션을 DB에 저장하고 active 상태로 전환.
   * 2개의 미션 카드 중 사용자가 원하는 것만 수락 가능.
   * aiMessage도 active에 함께 저장하여 중앙 패널에서 유지 표시.
   *
   * @param {Object} missionObj - 수락한 미션 객체
   */
  const handleAccept = async (missionObj) => {
    if (!preview) return;
    try {
      // POST /api/missions: 선택한 미션을 history 테이블에 삽입
      const res = await axios.post(`${API_BASE}/api/missions`, {
        mood, energy, outing: outingValue, comment,
        mission:       missionObj.mission,
        missionDetail: missionObj.missionDetail,
        energyTag:     missionObj.energyTag,
        policyName:    missionObj.policyName,
        policyLink:    missionObj.policyLink,
      });
      // active에 DB id + aiMessage 포함하여 저장
      setActive({ ...missionObj, id: res.data.id, aiMessage: preview.aiMessage });
      setPreview(null); // 미리보기 닫기
      fetchData();      // 통계/히스토리 갱신
    } catch (e) {
      alert('오류가 발생했습니다.');
    }
  };

  // ── 미션 완료 처리 ────────────────────────────────────────────────────────

  /**
   * 진행 중인 미션을 완료 처리.
   * POST /api/complete → history.is_completed = 1 업데이트
   * 완료 후 active 초기화 + 통계/히스토리 갱신
   */
  const handleComplete = async () => {
    if (!active?.id) return;
    try {
      await axios.post(`${API_BASE}/api/complete`, { id: active.id });
      setActive(null); // 완료된 미션 카드 닫기
      fetchData();     // streak/weekly 통계 갱신
    } catch (e) {
      alert('완료 처리 중 오류가 발생했습니다.');
    }
  };

  // ── 파생 값 계산 ──────────────────────────────────────────────────────────

  // 헤더 우측 상태 태그 (기분·에너지·외출 입력 시 자동 계산)
  const stateTags = mood ? computeStateTags(mood, energy, outingValue) : [];

  // AI 위로 메시지: preview 중이면 preview의 것, active면 active에 저장된 것 표시
  const aiMessage = preview?.aiMessage || active?.aiMessage;

  // ══════════════════════════════════════════════════════════════════════════
  // 렌더링
  // ══════════════════════════════════════════════════════════════════════════
  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#FFF5F3', fontFamily: 'system-ui, -apple-system, sans-serif' }}>

      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 헤더 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      <div style={{
        backgroundColor: 'white', borderBottom: '1px solid #F0E8FF',
        padding: '14px 32px', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        boxShadow: '0 1px 6px rgba(108,92,231,0.08)',
      }}>
        {/* 앱 타이틀 */}
        <div>
          <div style={{ fontSize: '11px', color: '#bbb', letterSpacing: '0.5px' }}>AI 기반 일상 행동 도움 앱</div>
          <div style={{ fontSize: '20px', fontWeight: '800', color: '#6C5CE7' }}>🐰 뛰었음 청년</div>
        </div>

        {/* 우측: 상태 태그 chips + streak + 회복 단계 배지 */}
        <div style={{ display: 'flex', gap: '6px', alignItems: 'center', flexWrap: 'wrap' }}>
          {/* 입력 기반 자동 상태 태그 */}
          {stateTags.map((t, i) => (
            <span key={i} style={{
              padding: '4px 10px', borderRadius: '20px', fontSize: '12px',
              backgroundColor: t.bg, color: t.color, fontWeight: 'bold',
            }}>{t.label}</span>
          ))}

          {/* 연속 달성 streak 배지 */}
          {stats?.streak > 0 && (
            <span style={{
              padding: '4px 12px', borderRadius: '20px', fontSize: '12px',
              backgroundColor: '#EDE7FF', color: '#6C5CE7', fontWeight: 'bold',
            }}>
              🔥 {stats.streak}일 연속
            </span>
          )}

          {/* 회복 단계 배지 (백엔드 규칙 기반 분류: 초기/중간/확장) */}
          {analysis?.stage && (() => {
            const s = STAGE_STYLES[analysis.stage] || {};
            return (
              <span style={{
                padding: '4px 12px', borderRadius: '20px', fontSize: '12px',
                backgroundColor: s.bg, color: s.color, fontWeight: 'bold',
                border: `1px solid ${s.color}33`,
              }}>
                {s.icon} {analysis.stage} 단계
              </span>
            );
          })()}
        </div>
      </div>

      {/* 서브타이틀 */}
      <div style={{ textAlign: 'center', padding: '12px 16px 4px', color: '#999', fontSize: '13px' }}>
        쉬었음을 비난하지 않고, 다시 움직일 수 있는 <strong style={{ color: '#6C5CE7' }}>가장 작은 첫걸음</strong>을 설계합니다.
      </div>

      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3단 컬럼 메인 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '290px 1fr 270px', // 좌:고정 / 중:유동 / 우:고정
        gap: '20px',
        padding: '20px 32px 40px',
        maxWidth: '1200px',
        margin: '0 auto',
      }}>

        {/* ═══════════════════════════ 좌측 패널: 상태 입력 ═══════════════════════════ */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {/* 상태 입력 카드 */}
          <div style={{ backgroundColor: 'white', borderRadius: '16px', padding: '22px', boxShadow: '0 2px 12px rgba(0,0,0,0.06)' }}>
            <h3 style={{ margin: '0 0 18px', fontSize: '15px', color: '#333', fontWeight: '700' }}>📋 오늘의 상태 입력</h3>

            {/* 기분 선택 버튼 (3종: 우울해/보통/편안함) */}
            <div style={{ marginBottom: '18px' }}>
              <label style={{ display: 'block', marginBottom: '10px', fontSize: '13px', color: '#666', fontWeight: '600' }}>
                오늘 기분은요?
              </label>
              <div style={{ display: 'flex', gap: '8px' }}>
                {MOOD_OPTIONS.map(opt => (
                  <button
                    key={opt.value}
                    onClick={() => setMood(opt.value)}
                    style={{
                      flex: 1, padding: '10px 4px', borderRadius: '12px',
                      // 선택된 버튼: 진한 배경 + 색상 테두리, 미선택: 연한 배경
                      border: `2px solid ${mood === opt.value ? opt.border : '#eee'}`,
                      backgroundColor: mood === opt.value ? opt.activeBg : opt.bg,
                      cursor: 'pointer', fontSize: '11px', textAlign: 'center',
                      fontWeight: mood === opt.value ? '700' : '400',
                      transition: 'all 0.15s', outline: 'none',
                    }}
                  >
                    <div style={{ fontSize: '22px', marginBottom: '4px' }}>{opt.emoji}</div>
                    {opt.value}
                  </button>
                ))}
              </div>
            </div>

            {/* 에너지 수준 슬라이더 (0~100) */}
            <div style={{ marginBottom: '18px' }}>
              <label style={{ display: 'block', marginBottom: '6px', fontSize: '13px', color: '#666', fontWeight: '600' }}>
                ⚡ 에너지 수준: <strong style={{ color: '#6C5CE7' }}>{energy}%</strong>
              </label>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '11px', color: '#bbb', minWidth: '30px' }}>낮아요</span>
                <input
                  type="range" min="0" max="100" value={energy}
                  onChange={e => setEnergy(Number(e.target.value))}
                  style={{ flex: 1, accentColor: '#6C5CE7', cursor: 'pointer' }}
                />
                <span style={{ fontSize: '11px', color: '#bbb', minWidth: '30px', textAlign: 'right' }}>높아요</span>
              </div>
            </div>

            {/* 외출 가능 정도 드롭다운 (4단계) */}
            <div style={{ marginBottom: '18px' }}>
              <label style={{ display: 'block', marginBottom: '6px', fontSize: '13px', color: '#666', fontWeight: '600' }}>
                🚶 외출 가능 정도
              </label>
              <select
                value={outingValue}
                onChange={e => setOutingValue(Number(e.target.value))}
                style={{
                  width: '100%', padding: '10px 12px', borderRadius: '10px',
                  border: '1px solid #E0E0E0', fontSize: '13px', backgroundColor: 'white',
                  cursor: 'pointer', color: '#444', outline: 'none',
                }}
              >
                {OUTING_OPTIONS.map(opt => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
            </div>

            {/* 오늘의 한마디 자유 텍스트 입력 (선택사항, GPT 쿼리 확장에 활용) */}
            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', marginBottom: '6px', fontSize: '13px', color: '#666', fontWeight: '600' }}>
                💬 오늘의 한마디 <span style={{ fontWeight: '400', color: '#bbb' }}>(선택)</span>
              </label>
              <textarea
                placeholder="오늘 어떤 하루인가요? 자유롭게 적어주세요."
                value={comment}
                onChange={e => setComment(e.target.value)}
                style={{
                  width: '100%', height: '68px', padding: '10px', boxSizing: 'border-box',
                  borderRadius: '10px', border: '1px solid #E0E0E0', fontSize: '13px',
                  resize: 'none', outline: 'none', color: '#444', lineHeight: '1.5',
                }}
              />
            </div>

            {/* 미션 추천 버튼 (RAG + GPT 파이프라인 호출) */}
            <button
              onClick={fetchMission}
              disabled={loading || !mood} // 기분 미선택 또는 로딩 중 비활성화
              style={{
                width: '100%', padding: '14px', borderRadius: '12px', border: 'none',
                backgroundColor: !mood ? '#D0D0D0' : loading ? '#a89fe8' : '#6C5CE7',
                color: 'white', fontSize: '15px', fontWeight: '700',
                cursor: (!mood || loading) ? 'not-allowed' : 'pointer',
                transition: 'background-color 0.2s',
              }}
            >
              {loading ? '⏳ AI 분석 중...' : '✨ 미션 추천받기'}
            </button>
          </div>

          {/* 통계 카드: streak + 최근 완료 + 분석 인사이트 */}
          <div style={{ backgroundColor: 'white', borderRadius: '16px', padding: '18px 22px', boxShadow: '0 2px 12px rgba(0,0,0,0.06)' }}>
            {/* 상단: 연속 달성일 + 최근 완료 미션 */}
            <div style={{ display: 'flex', gap: '12px', marginBottom: analysis?.totalMissions > 0 ? '12px' : '0' }}>
              <div style={{ flex: 1, textAlign: 'center', padding: '12px 8px', backgroundColor: '#EDE7FF', borderRadius: '12px' }}>
                <div style={{ fontSize: '28px', fontWeight: '800', color: '#6C5CE7', lineHeight: 1 }}>
                  {stats?.streak ?? 0}
                </div>
                <div style={{ fontSize: '10px', color: '#9B8FCF', marginTop: '5px', fontWeight: '600' }}>일 연속 달성</div>
              </div>
              <div style={{ flex: 1.6, padding: '12px 10px', backgroundColor: '#FFF3F3', borderRadius: '12px' }}>
                <div style={{ fontSize: '10px', color: '#F0A0A0', fontWeight: '600', marginBottom: '5px' }}>최근 완료 미션</div>
                <div style={{ fontSize: '12px', fontWeight: '700', color: '#E87E79', lineHeight: '1.4' }}>
                  {stats?.lastMission
                    ? stats.lastMission.slice(0, 18) + (stats.lastMission.length > 18 ? '…' : '')
                    : '아직 없어요'}
                </div>
              </div>
            </div>

            {/* 하단: numpy/sklearn 분석 결과 인사이트 (미션 기록이 있을 때만 표시) */}
            {analysis && analysis.totalMissions > 0 && (
              <div style={{ borderTop: '1px solid #F5F5F5', paddingTop: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {/* 전체 완료율 progress bar */}
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                    <span style={{ fontSize: '10px', color: '#999', fontWeight: '600' }}>전체 완료율</span>
                    <span style={{ fontSize: '10px', color: '#6C5CE7', fontWeight: '700' }}>
                      {Math.round(analysis.completionRate * 100)}%
                    </span>
                  </div>
                  {/* numpy mean()으로 계산된 완료율을 시각화 */}
                  <div style={{ height: '6px', backgroundColor: '#F0F0F0', borderRadius: '4px', overflow: 'hidden' }}>
                    <div style={{
                      height: '100%', borderRadius: '4px', backgroundColor: '#6C5CE7',
                      width: `${Math.round(analysis.completionRate * 100)}%`,
                      transition: 'width 0.5s ease',
                    }} />
                  </div>
                </div>

                {/* 성공 시 평균 에너지·외출 (numpy 분석 결과) */}
                {analysis.avgEnergyOnSuccess > 0 && (
                  <div style={{ fontSize: '11px', color: '#888', lineHeight: '1.5' }}>
                    💡 완료 시 평균: 에너지 {analysis.avgEnergyOnSuccess}%,
                    외출 {analysis.avgOutingOnSuccess}%
                  </div>
                )}

                {/* 완료율 가장 높은 기분 (기분별 완료율 중 max) */}
                {analysis.bestMood && (
                  <div style={{ fontSize: '11px', color: '#888' }}>
                    ✨ 가장 잘 되는 기분: <strong style={{ color: '#6C5CE7' }}>{analysis.bestMood}</strong>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* ═══════════════════════════ 중앙 패널: AI 미션 ═══════════════════════════ */}
        <div style={{
          backgroundColor: 'white', borderRadius: '16px', padding: '24px',
          boxShadow: '0 2px 12px rgba(0,0,0,0.06)', minHeight: '480px',
          display: 'flex', flexDirection: 'column',
        }}>
          <div>
            <h3 style={{ margin: '0 0 2px', fontSize: '16px', color: '#333', fontWeight: '700' }}>
              🎯 AI 맞춤형 회복 미션
            </h3>
            <p style={{ margin: 0, fontSize: '12px', color: '#bbb' }}>추천 기준: 상태 + 이력 + 최근 성공 패턴</p>
          </div>
          <div style={{ borderTop: '1px solid #F5F5F5', margin: '16px 0' }} />

          {/* 초기 안내 (기분 미입력 또는 추천 전) */}
          {!preview && !active && !loading && (
            <div style={{
              flex: 1, display: 'flex', flexDirection: 'column',
              alignItems: 'center', justifyContent: 'center', color: '#D0D0D0', padding: '20px',
            }}>
              <div style={{ fontSize: '56px', marginBottom: '16px' }}>🐰</div>
              <p style={{ textAlign: 'center', lineHeight: '1.7', fontSize: '14px' }}>
                왼쪽에서 오늘의 상태를 입력하고<br />
                <strong style={{ color: '#6C5CE7' }}>미션 추천받기</strong>를 눌러보세요!
              </p>
            </div>
          )}

          {/* AI 응답 대기 중 */}
          {loading && (
            <div style={{
              flex: 1, display: 'flex', flexDirection: 'column',
              alignItems: 'center', justifyContent: 'center', color: '#999',
            }}>
              <div style={{ fontSize: '40px', marginBottom: '12px' }}>⏳</div>
              <p style={{ fontSize: '14px' }}>AI가 딱 맞는 미션을 찾고 있어요...</p>
            </div>
          )}

          {/* GPT 생성 위로 메시지 (preview와 active 모두에서 표시) */}
          {aiMessage && !loading && (
            <div style={{
              backgroundColor: '#F3EEFF', padding: '16px 18px', borderRadius: '12px',
              borderLeft: '4px solid #6C5CE7', marginBottom: '16px',
            }}>
              <p style={{ margin: 0, fontSize: '14px', lineHeight: '1.75', color: '#555' }}>
                💌 {aiMessage}
              </p>
            </div>
          )}

          {/* preview 모드: 2개 미션 카드 동시 표시 (사용자가 원하는 것 선택) */}
          {preview && !loading && (
            <div style={{ overflowY: 'auto', flex: 1 }}>
              {(preview.missions || []).map((mission, idx) => (
                <MissionCard
                  key={idx}
                  mission={mission}
                  isActive={false}
                  onAccept={() => handleAccept(mission)} // 이 카드의 미션만 수락
                  onRefresh={fetchMission}
                  loading={loading}
                />
              ))}
            </div>
          )}

          {/* active 모드: 수락 확정된 미션 1개 표시 */}
          {active && !loading && (
            <MissionCard
              mission={active}
              isActive={true}
              onComplete={handleComplete}
            />
          )}
        </div>

        {/* ═══════════════════════════ 우측 패널: 회복 기록 ═══════════════════════════ */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={{ backgroundColor: 'white', borderRadius: '16px', padding: '22px', boxShadow: '0 2px 12px rgba(0,0,0,0.06)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h3 style={{ margin: 0, fontSize: '15px', color: '#333', fontWeight: '700' }}>📅 회복 기록</h3>
              {/* 수동 새로고침 (미션 완료 후 즉시 반영 확인용) */}
              <button
                onClick={fetchData}
                style={{ border: 'none', background: 'none', cursor: 'pointer', color: '#6C5CE7', fontSize: '12px', fontWeight: '600' }}
              >
                새로고침
              </button>
            </div>

            {/* 주간 캘린더 (월~일 완료 현황 원형 아이콘) */}
            <WeeklyCalendar weekly={stats?.weekly} />

            {/* streak 달성 배너 */}
            {stats?.streak > 0 && (
              <div style={{
                textAlign: 'center', backgroundColor: '#EDE7FF',
                borderRadius: '10px', padding: '10px', marginBottom: '16px',
              }}>
                <span style={{ color: '#6C5CE7', fontWeight: '700', fontSize: '13px' }}>
                  🔥 {stats.streak}일 연속 미션 완료 중!
                </span>
              </div>
            )}

            {/* K-Means 클러스터링 분석 결과 (scikit-learn, 5개 이상 데이터 시 활성화) */}
            {analysis?.clusterInsights && (
              <div style={{ backgroundColor: '#F8F5FF', borderRadius: '12px', padding: '14px', marginBottom: '16px' }}>
                <p style={{ margin: '0 0 8px', fontSize: '11px', color: '#6C5CE7', fontWeight: '700' }}>
                  🤖 AI 행동 패턴 분석
                </p>
                {/* 분석 방법 설명 (에너지×외출 K-Means) */}
                <p style={{ margin: '0 0 8px', fontSize: '10px', color: '#999' }}>
                  {analysis.clusterInsights.description}
                </p>
                {/* 각 클러스터별 이름, 중심값, 완료율 */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
                  {analysis.clusterInsights.names.map((name, i) => {
                    const [e, o] = analysis.clusterInsights.centers[i];
                    const rate   = analysis.clusterInsights.completionRates[i];
                    return (
                      <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div>
                          <span style={{ fontSize: '11px', fontWeight: '600', color: '#555' }}>{name}</span>
                          {/* 클러스터 중심의 에너지·외출 수준 (원래 스케일로 역변환된 값) */}
                          <span style={{ fontSize: '10px', color: '#bbb', marginLeft: '6px' }}>
                            E:{Math.round(e)}% O:{Math.round(o)}%
                          </span>
                        </div>
                        {/* 클러스터별 완료율 (색상 코딩: 60%↑초록, 30~60%노랑, 30%↓빨강) */}
                        <span style={{
                          fontSize: '10px', fontWeight: '700', padding: '2px 7px', borderRadius: '8px',
                          backgroundColor: rate >= 0.6 ? '#D4EDDA' : rate >= 0.3 ? '#FFF3CD' : '#FFE0E0',
                          color: rate >= 0.6 ? '#155724' : rate >= 0.3 ? '#856404' : '#C0392B',
                        }}>
                          완료율 {Math.round(rate * 100)}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* 미션 이력 리스트 (최근 30개, 스크롤 가능) */}
            <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
              {history.length === 0 ? (
                <p style={{ color: '#ccc', textAlign: 'center', fontSize: '13px', padding: '24px 0' }}>
                  아직 기록이 없어요<br />첫 미션을 시작해보세요!
                </p>
              ) : (
                history.map(item => (
                  <div key={item.id} style={{ padding: '11px 0', borderBottom: '1px solid #F5F5F5' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '8px' }}>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        {/* 완료된 미션은 보라색, 진행 중은 기본색 */}
                        <p style={{
                          margin: '0 0 4px', fontSize: '13px', fontWeight: '600',
                          color: item.isCompleted ? '#6C5CE7' : '#333',
                          whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                        }}>
                          {item.mission}
                        </p>
                        {/* 메타 정보: 날짜 + 기분 + 난이도 뱃지 */}
                        <div style={{ display: 'flex', gap: '5px', alignItems: 'center', flexWrap: 'wrap' }}>
                          <span style={{ fontSize: '11px', color: '#bbb' }}>{item.date}</span>
                          <span style={{ fontSize: '11px', color: '#ddd' }}>·</span>
                          <span style={{ fontSize: '11px', color: '#bbb' }}>{item.mood}</span>
                          {/* energyTag 컬러 뱃지 (ENERGY_TAG_STYLES 기반) */}
                          {item.energyTag && (
                            <span style={{
                              fontSize: '10px', padding: '1px 7px', borderRadius: '8px', fontWeight: '600',
                              backgroundColor: (ENERGY_TAG_STYLES[item.energyTag] || {}).bg || '#F0F0F0',
                              color: (ENERGY_TAG_STYLES[item.energyTag] || {}).color || '#666',
                            }}>
                              {item.energyTag}
                            </span>
                          )}
                        </div>
                      </div>
                      {/* 완료/진행중 상태 뱃지 */}
                      <span style={{
                        padding: '3px 9px', borderRadius: '20px', fontSize: '11px',
                        whiteSpace: 'nowrap', fontWeight: '600',
                        backgroundColor: item.isCompleted ? '#EDE7FF' : '#FFF0F0',
                        color: item.isCompleted ? '#6C5CE7' : '#E74C3C',
                      }}>
                        {item.isCompleted ? '✔ 완료' : '진행중'}
                      </span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
