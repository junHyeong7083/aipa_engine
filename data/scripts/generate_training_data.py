"""
AIPA-Eval 모델 학습 데이터 생성 스크립트

C#으로 비유하면 배치 프로그램으로 AI에게 대량의 학습 데이터를 생성시키는 도구.
Claude API(선생님)에게 "이 페르소나가 이 자극물을 보면 어떻게 평가할까?" 질문을
수천 번 반복해서 학습 데이터를 만듦.

이렇게 만든 데이터로 Qwen 2.5-7B를 LoRA 파인튜닝 → AIPA-Eval 모델 완성.
이 과정을 "Knowledge Distillation (지식 증류)"이라고 함.

사용법:
    python data/scripts/generate_training_data.py --count 100           # 100건 테스트
    python data/scripts/generate_training_data.py --count 3000 --concurrent 10  # 3000건 본격 생성

필요 환경변수:
    ANTHROPIC_API_KEY=your_key
"""

import asyncio
import json
import re
import os
import random
import argparse
from pathlib import Path
from datetime import datetime

# .env 파일에서 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

# anthropic = Claude API Python SDK
import anthropic

# ─────────────────────────────────────────────
# 자극물 시나리오 (카테고리별)
# 학습 데이터의 다양성을 위해 16개 카테고리 × 다양한 시나리오 확보
# C#의 Dictionary<string, List<string>> 같은 구조
# ─────────────────────────────────────────────

STIMULI = {
    "식품": [
        "새로운 프리미엄 치킨 메뉴 '허니갈릭 치킨'. 가격 18,000원. 매콤달콤한 허니갈릭 소스에 후라이드 치킨을 버무린 메뉴. 사이드로 치즈볼 2개 포함.",
        "편의점 신상 도시락 '한우 불고기 도시락'. 가격 6,900원. 한우 등급 불고기와 계란찜, 김치, 시금치 나물 구성.",
        "배달 전문 브랜드 '건강한 한 끼' 샐러드 구독 서비스. 월 89,000원에 주 5회 배달. 매일 다른 토핑과 드레싱.",
        "스타벅스 시즌 한정 음료 '제주 한라봉 프라푸치노'. 가격 6,500원. 제주 한라봉 과즙 + 크림 + 한라봉 껍질 토핑.",
        "무알콜 수제맥주 '제로비어 IPA'. 500ml 캔, 가격 3,500원. 일반 IPA와 동일한 홉 향미, 칼로리 30kcal.",
        "프리미엄 컵밥 '직화 불닭 컵밥'. 가격 4,500원. 매운맛 5단계 중 3단계. 전자레인지 3분. 불닭소스 + 치즈 토핑.",
        "비건 레스토랑 '그린테이블' 코스 메뉴. 가격 45,000원. 전채, 메인(콩고기 스테이크), 디저트 3코스. 예약제.",
        "밀키트 '30분 완성 감바스'. 가격 12,900원. 새우, 마늘, 올리브오일, 바게트 포함. 2인분.",
        "다이어트 간식 '단백질 브라우니'. 1박스(10개) 19,800원. 개당 단백질 15g, 설탕 제로. 초코/말차 2종.",
        "로컬 카페 브랜드 '동네 한 잔'. 아메리카노 2,500원. 스페셜티 원두 사용. 텀블러 지참 시 500원 할인.",
    ],
    "화장품": [
        "비건 선크림 'CICA Sun Shield SPF50+'. 50ml, 가격 28,000원. 병풀 추출물 함유, 백탁 없음, 민감성 피부 테스트 완료.",
        "올리브영 신상 립틴트 '무드 글로우 틴트'. 가격 15,000원. 12시간 지속력, 보습 기능, 5가지 색상.",
        "남성용 올인원 스킨케어 '맨즈 데일리 에센스'. 150ml, 가격 22,000원. 세안 후 하나로 끝. 무향, 끈적임 없음.",
        "프리미엄 헤어 에센스 '실크드롭'. 100ml, 가격 35,000원. 아르간오일 + 케라틴. 열 보호 기능. 향수 대용 가능한 향.",
        "클린뷰티 파운데이션 '스킨퓨어 쿠션'. 가격 32,000원. 12가지 유해성분 프리. 피부톤 보정 + 선케어. 리필 포함.",
        "탈모 방지 샴푸 '헤어풀 스칼프 클렌저'. 500ml, 가격 25,000원. 탈모증상완화 기능성 인증. 쿨링 멘톨 함유.",
        "레티놀 안티에이징 세럼 '리뉴셀 0.3%'. 30ml, 가격 42,000원. 캡슐형 레티놀. 자극 최소화 공법. 피부과 테스트 완료. 밤 전용.",
        "쿠션팩트 '글로우핏 메쉬쿠션'. 가격 38,000원. 메쉬 망 구조로 소량씩 배출. 12시간 밀착. 21호/23호/25호. SPF50+.",
        "바디로션 '세라마이드 딥모이스처'. 400ml, 가격 18,000원. 세라마이드 3종 + 히알루론산. 무향료. 아토피 피부 사용 가능.",
        "컬러 립밤 '멜팅 틴티드 밤'. 가격 12,000원. 발색 + 보습 동시에. 시어버터 함유. 4색상. 동물실험 프리.",
    ],
    "앱/서비스": [
        "AI 기반 가계부 앱 '머니로그'. 월 4,900원. 영수증 사진 찍으면 자동 분류. 소비 패턴 분석 + 절약 팁 제공.",
        "동네 산책 매칭 앱 '걷자'. 무료. 같은 동네 비슷한 시간대 산책하는 사람끼리 매칭. 안전 기능(실시간 위치 공유) 포함.",
        "구독형 독서 요약 서비스 '3분 독서'. 월 9,900원. 매주 베스트셀러 3권을 10분 분량으로 요약. 오디오 + 텍스트 제공.",
        "중고 명품 감정 앱 '리얼체크'. 건당 5,000원. AI + 전문 감정사 이중 검증. 정품 인증서 발급. 감정 소요 24시간.",
        "수면 관리 앱 '슬립케어'. 월 3,900원. 수면 패턴 AI 분석 + 맞춤 백색소음 + 스마트 알람. 애플워치 연동.",
        "반려동물 돌봄 매칭 '펫시터'. 시간당 15,000원. 검증된 펫시터 매칭. 실시간 사진 리포트. 보험 가입 포함.",
        "AI 영어 회화 앱 '토키'. 월 14,900원. AI와 음성 대화. 발음 교정 + 표현 추천. 상황별 시나리오 100개+.",
    ],
    "광고": [
        "삼성 갤럭시 Z 플립 광고. 20대 여성 인플루언서가 카페에서 플립을 접어 셀카 찍는 15초 영상. BGM은 뉴진스 노래. 카피: '접으면 달라지는 나'.",
        "현대자동차 아이오닉 6 광고. 한적한 해안도로를 달리는 30초 영상. '0원의 주유비, 100%의 드라이빙' 카피. 차량 내부 인테리어 강조.",
        "당근마켓 TV 광고. 이웃 간 중고거래 에피소드 30초. 할머니가 손녀에게 중고 자전거를 당근에서 구해주는 스토리.",
        "쿠팡 로켓배송 광고. 밤 11시 주문 → 새벽 도착. 잠옷 입은 남성이 새벽에 문 앞 택배 열어보며 감동하는 15초.",
        "네이버 하이퍼클로바X 광고. 직장인이 AI에게 보고서 초안을 부탁하는 30초. '일은 AI가, 퇴근은 내가' 카피.",
        "토스 광고. 복잡한 송금을 한 번 터치로 끝내는 15초. '금융을 쉽게' 카피. 미니멀한 UI 화면 클로즈업.",
    ],
    "보험/금융": [
        "MZ세대 맞춤 보험 '마이 미니 보험'. 월 5,000원. 입원비 + 실손 + 펫보험 중 원하는 것만 선택. 앱에서 1분 가입.",
        "카카오뱅크 신규 적금 상품 '26주 챌린지 적금'. 매주 1만원씩 26주. 연 4.5% 금리. 목표 달성 시 추가 0.5% 보너스.",
        "토스 주식 모으기 서비스. 매일 1,000원씩 자동 분산 투자. S&P500, 국내 우량주 등 포트폴리오 5종. 수수료 무료.",
        "신한은행 전세대출 '올인원 전세론'. 최대 3억, 연 3.8%. 비대면 신청 가능. 심사 최단 1일. 중도상환 수수료 없음.",
        "KB국민은행 주택담보대출 '내집마련 디딤돌'. 최대 2.5억, 연 3.2%. 신혼부부 우대금리 0.2%p. 40년 만기. 원리금균등상환.",
        "삼성화재 운전자보험 '착한 운전자'. 월 12,000원. 교통사고 처리지원 + 벌금 + 면허정지 위로금. 자녀할인 10%.",
        "미래에셋 연금저축펀드 'TDF 2050'. 연 수익률 8.2%(3년 평균). 자동 자산배분. 세액공제 최대 99만원. 월 자동이체 가능.",
        "카카오페이 후불결제 '카카오페이 나중에결제'. 최대 30만원. 다음 달 일괄 상환. 수수료 무료. 신용점수 영향 없음.",
    ],
    "콘텐츠": [
        "넷플릭스 한국 오리지널 드라마 '서울 좀비'. 회사원들이 야근 중 좀비 사태를 맞는 블랙코미디. 8부작. 주연: 마동석, 한소희.",
        "유튜브 채널 '과학쿠키' 새 시리즈 '일상 속 물리학'. 10분 영상, 주 2회. '엘리베이터에서 왜 어색한가'같은 주제를 물리학으로 풀어냄.",
        "웹소설 '회귀한 S급 헌터'. 매일 1화 무료, 이후 화당 300원. 판타지/액션. 현재 150화 연재 중. 평점 9.2.",
        "팟캐스트 '퇴사한 형들의 창업 일지'. 매주 수요일 30분. 실제 창업 실패/성공 경험담. 스타트업 대표 게스트 출연.",
        "유튜브 다큐 시리즈 '한국의 맛'. 전국 로컬 맛집 탐방 20분. 셰프 출신 유튜버가 조리 과정 분석. 주 1회 업로드.",
    ],
    "사업계획서": [
        "반려동물 건강관리 플랫폼 '펫닥'. 수의사 비대면 상담 + AI 건강 체크 + 맞춤 사료 추천. 월 구독 29,000원. 타겟: 2030 반려인.",
        "시니어 디지털 교육 플랫폼 '쉽게 배우는 스마트폰'. 60대 이상 대상. 월 15,000원. 1:1 화상 레슨 + 따라하기 영상.",
        "공유 주방 플랫폼 '쿡셰어'. 시간당 15,000원. 소규모 외식 창업자/배달 전문점용. 장비+공간+배달앱 연동 원스톱.",
        "AI 이력서 매칭 서비스 '핏잡'. 이력서 등록하면 AI가 적합 공고 자동 추천 + 합격률 예측. 구직자 무료, 기업 월 50만원.",
        "제로웨이스트 생활용품 구독 '지구샵 박스'. 월 29,900원. 친환경 세제/칫솔/수세미 등 매월 배송. 빈 용기 회수 시 할인.",
    ],
    "이벤트/프로모션": [
        "올리브영 '올영세일'. 전 품목 최대 50% 할인, 3일간. 앱 전용 추가 10% 쿠폰. 5만원 이상 구매 시 랜덤박스 증정.",
        "스타벅스 여름 e-프리퀀시 이벤트. 음료 17잔 구매 시 서머 캐리백 증정. 미션음료 3잔 포함 필수.",
        "무신사 블랙프라이데이. 최대 80% 할인. 선착순 한정 1만원 쿠폰. 10만원 이상 구매 시 무료배송 + 사은품.",
        "네이버플러스 멤버십 첫 달 100원 이벤트. 이후 월 4,900원. 네이버페이 포인트 4% 적립 + 넷플릭스 등 OTT 할인.",
        "배달의민족 '한집배달 무료배달' 이벤트. 2주간 한집배달 주문 시 배달비 0원. 최소주문 15,000원.",
    ],
    "패션": [
        "유니클로 '에어리즘 코튼 오버사이즈 티'. 가격 19,900원. 냉감 소재 + 면 혼방. 남녀공용. 6색상.",
        "무신사 스탠다드 와이드 데님 팬츠. 가격 39,900원. 13.5oz 셀비지 데님. 원워시. S/M/L/XL.",
        "나이키 에어맥스 DN. 가격 179,000원. 새로운 에어 유닛 '다이나믹 에어' 기술. 런닝/일상 겸용.",
        "국내 브랜드 '마뗑킴' 크로셰 니트백. 가격 68,000원. 핸드메이드. 3색상. 올여름 트렌드 아이템.",
        "아디다스 삼바 OG. 가격 139,000원. 클래식 레트로 디자인. 천연 가죽 + 스웨이드. 화이트/블랙 2색상.",
        "한국 브랜드 '디스이즈네버댓' 나일론 숄더백. 가격 89,000원. 방수 소재. 크로스바디 가능. 남녀공용.",
        "자라 리넨 블렌드 셔츠. 가격 59,900원. 린넨 55% + 면 45%. 오버핏. 여름 시즌 한정. 5색상.",
        "MLB 모노그램 볼캡. 가격 45,000원. 뉴욕양키스 로고. 면 100%. 사이즈 조절 가능. 6색상.",
    ],
    "가전/전자": [
        "삼성 비스포크 냉장고 4도어. 가격 250만원. 맞춤 색상 패널. 정온 기술. 849L. AI 절전 모드.",
        "다이슨 에어랩 멀티 스타일러. 가격 69만원. 6종 헤드. 코안다 기술. 열 손상 최소화.",
        "LG 스탠바이미 2세대. 가격 109만원. 27인치 이동식 터치스크린. 배터리 3시간. 넷플릭스/유튜브 내장.",
        "애플 비전프로. 가격 499만원. 공간 컴퓨팅. AR/VR. 아이트래킹 + 핸드 제스처 조작.",
        "로봇청소기 '로보락 S8 맥스V 울트라'. 가격 189만원. 걸레 자동세척 + 자동비움 + 뜨거운 바람 건조. 장애물 인식 AI.",
        "삼성 갤럭시 버즈3 프로. 가격 329,000원. ANC(능동소음제거). 24bit Hi-Fi. 통화 소음 제거. 배터리 7시간.",
        "LG 퓨리케어 360 공기청정기. 가격 89만원. 360도 흡입. 클린부스터. 68㎡ 커버. 스마트씽큐 연동.",
        "아이패드 에어 M3. 가격 899,000원. 11인치 Liquid Retina. Apple Pencil Pro 지원. 128GB. Wi-Fi 6E.",
    ],
    "교육": [
        "AI 과외 플랫폼 '튜터봇'. 월 49,000원. GPT 기반 1:1 수학 과외. 풀이 과정 단계별 설명. 오답노트 자동 생성. 중·고등 수학 전 과정.",
        "성인 영어 회화 학원 '스픽이지'. 월 198,000원. 원어민 소그룹(4인) 수업. 주 3회 50분. 강남역 도보 3분. 레벨 테스트 무료.",
        "코딩 부트캠프 '제로베이스 백엔드 스쿨'. 6개월 과정, 총 450만원. Java/Spring 중심. 취업 연계율 87%. 국비지원 시 자부담 50만원.",
        "온라인 자격증 강의 '에듀윌 공인중개사'. 12개월 패키지 89만원. 기본+심화+모의고사. 합격 시 수강료 50% 환급.",
        "유아 영어 앱 '링고키즈'. 월 12,900원. 게임형 학습. 파닉스 + 단어 + 문장. 3~8세 대상. 광고 없음.",
        "EBS 수능 인터넷 강의 '수능특강 국어'. 무료. 국어 영역 전 범위. 강의당 40분. 교재 별도 8,900원.",
        "직장인 MBA 과정 '서울대 경영전문대학원 EMBA'. 2년 과정 총 6,000만원. 주말 수업. 해외 교환 프로그램 포함.",
        "초등학생 수학 학습지 '쎈수학'. 월 35,000원. 유형별 문제풀이. 주 2회 방문 첨삭. 초1~초6 전 과정.",
        "드로잉 온라인 클래스 '클래스101 인물화 마스터'. 149,000원(평생 소장). 20시간 분량. 초보~중급. 재료키트 포함.",
    ],
    "정책": [
        "청년 월세 지원 정책 '청년월세 한시 특별지원'. 월 최대 20만원, 12개월. 만 19~34세. 소득 중위 60% 이하. 보증금 5천만원 이하.",
        "국민연금 개편안. 보험료율 9%→13% 인상. 소득대체율 40%→45%. 수급 개시 65세 유지. 2026년부터 단계적 적용.",
        "육아휴직 급여 인상안. 통상임금 80%→100%(상한 월 250만원). 부부 동시 육아휴직 허용. 최대 18개월.",
        "최저임금 2027년 인상안. 시급 10,030원→10,800원(7.7% 인상). 업종별 차등적용 논의 중. 소상공인 지원금 병행.",
        "신혼부부 특별공급 확대. 소득기준 도시근로자 140%까지. 자녀 수에 따라 가점. 수도권 공공분양 30% 배정.",
        "기초연금 인상안. 월 32만원→40만원. 만 65세 이상 소득 하위 70%. 2027년부터 시행. 재원: 일반회계.",
        "소상공인 전기요금 지원. 월 최대 20만원. 연매출 3천만원 이하. 6개월간. 한전 고지서 기준 자동 차감.",
        "대중교통 기후동행카드. 월 65,000원 정액제. 서울시 버스+지하철+따릉이 무제한. 경기·인천 연계 확대 검토 중.",
    ],
    "부동산": [
        "서울 마포구 '마포래미안 푸르지오' 아파트 34평. 매매가 14.5억. 역세권(마포역 도보 5분). 2019년 입주. 한강뷰 일부 세대.",
        "경기 성남시 '판교 알파리움' 오피스텔 전용 24평. 월세 보증금 3천만원/월 90만원. 판교역 도보 8분. 풀옵션.",
        "서울 강동구 '둔촌주공 재건축' 올림픽파크포레온. 분양가 34평 기준 12.8억. 1만2천세대 대단지. 2025년 입주.",
        "공유주거 서비스 '맹그로브'. 월 85만원(관리비 포함). 개인실 + 공용 라운지/주방. 강남 위치. 1~12개월 계약.",
        "제주시 애월읍 전원주택. 대지 200평 + 건물 40평. 매매가 5.8억. 바다 조망. 올수리 완료. 텃밭 가능.",
        "인천 송도 '랜드마크시티 센트럴파크' 아파트 25평. 전세 4.5억. 센트럴파크 뷰. 학군 우수. 2020년 입주.",
        "서울 영등포구 역세권 오피스텔 투자물건. 전용 15평. 매매가 2.3억. 월세 보증금 1천/월 75만원. 수익률 3.9%.",
        "경기 화성시 동탄2 '동탄역 롯데캐슬' 49평. 매매가 9.2억. 동탄역 SRT 이용 가능. 대형 커뮤니티 시설.",
        "서울 종로구 '익선동 한옥 리모델링 상가'. 보증금 5천만원/월 180만원. 15평. 카페/소매업 적합. 관광객 유동인구 다수.",
    ],
    "자동차": [
        "현대 아이오닉 5 N. 가격 7,380만원. 전기차. 650마력. 제로백 3.4초. 1회 충전 448km. N 드리프트 모드.",
        "기아 EV3. 가격 3,477만원~. 소형 전기 SUV. 1회 충전 501km. 15.5인치 디스플레이. ADAS 기본 탑재.",
        "테슬라 모델 Y 롱레인지. 가격 5,699만원. 1회 충전 511km. 오토파일럿 기본. OTA 업데이트.",
        "쏘카 '플랜' 자동차 구독. 월 599,000원~. 보험+정비 포함. 월 1회 차종 변경 가능. 6개월/12개월 약정.",
        "현대 싼타페 디 엣지. 가격 3,622만원~. 하이브리드. 복합연비 14.5km/L. 3열 시트. 파노라마 선루프.",
        "케이카 내차사기 홈서비스. 중고차 '현대 투싼 2022년식' 2,480만원. 무사고. 주행 3.2만km. 홈 배송 + 3일 환불보장.",
        "BMW X3 30e xDrive PHEV. 가격 7,290만원. 플러그인 하이브리드. 전기 모드 47km. 복합 292마력.",
        "르노 그랑 콜레오스. 가격 3,757만원~. 중형 SUV. 구글 빌트인. 2열 독립시트. 복합연비 12.3km/L.",
        "KG 모빌리티 토레스 EVX. 가격 3,930만원~. 전기 SUV. 1회 충전 462km. V2L(외부 전력 공급). 캠핑 모드.",
    ],
    "설문지": [
        "직장인 워라밸 실태조사. '귀하의 주당 평균 야근 시간은?' '현재 직장의 워라밸 만족도를 1~10으로 평가해주세요.' 총 15문항. 소요시간 5분.",
        "신제품 만족도 설문 '○○ 무선청소기 사용 후기'. '흡입력에 만족하십니까?' '가격 대비 성능은?' 총 10문항. 응답 시 스타벅스 쿠폰 증정.",
        "사회이슈 인식조사 '반려동물 문화'. '반려동물 등록제 의무화에 찬성하십니까?' '펫티켓 교육이 필요하다고 생각하십니까?' 총 12문항.",
        "대학생 식생활 조사. '하루 평균 외식 횟수는?' '한 끼 평균 식비는?' '건강한 식단에 관심이 있으십니까?' 총 20문항. 소요시간 7분.",
        "온라인 쇼핑 경험 설문. '가장 자주 이용하는 쇼핑 플랫폼은?' '구매 결정 시 가장 중요한 요소는?' 총 15문항. 추첨 100명 치킨 기프티콘.",
        "주거환경 만족도 조사. '현재 거주 형태는?' '월 주거비 부담을 1~10으로 평가해주세요.' '이사 계획이 있으십니까?' 총 18문항.",
        "MZ세대 투자 성향 조사. '현재 투자 중인 상품은?' '월 투자 금액은?' '투자 정보를 어디서 얻으십니까?' 총 14문항. 소요시간 6분.",
        "직장 내 AI 도구 활용 설문. 'AI 도구 사용 경험이 있으십니까?' '업무 생산성에 도움이 되었습니까?' 총 10문항. 소요시간 4분.",
    ],
    "기타": [
        "코워킹 스페이스 '패스트파이브 강남점'. 자유석 월 250,000원. 전용 데스크 월 350,000원. 회의실/폰부스/라운지 무료. 24시간 이용.",
        "웨딩 플래닝 서비스 '마이웨딩'. 패키지 350만원~. 스드메(스튜디오+드레스+메이크업) 원스톱. 본식 진행 포함. 평일 할인 30%.",
        "장례 서비스 '좋은장례'. 3일장 기준 350만원~. 장지+제단+수의+식사 포함. 24시간 상담. 사전 계약 시 10% 할인.",
        "포장이사 서비스 '짐싸'. 원룸 기준 35만원~. 포장+운송+정리 풀서비스. 보험 가입. 당일 견적. 후불제.",
        "반려동물 장례 서비스 '무지개다리'. 개별 화장 15만원~. 유골함+발도장 기본 제공. 추모 공간. 24시간 픽업.",
        "세탁 픽업 서비스 '런드리고'. 월 구독 39,900원. 주 1회 수거+배달. 셔츠/바지 등 10벌. 추가 건당 2,000원.",
        "키즈카페 '플레이즈 잠실점'. 2시간 기준 아동 18,000원, 보호자 무료. 300평 규모. 볼풀+트램폴린+공작실. 주말 예약 필수.",
        "시니어 돌봄 서비스 '케어닥'. 시간당 15,000원. 병원 동행+가사+말벗. 요양보호사 자격 검증. 정부 바우처 연계 가능.",
        "프리미엄 세차 서비스 '왁싱카'. 출장 세차 49,000원~. 내외부 클리닝+코팅. 친환경 무수세차. 앱 예약. 아파트 방문.",
    ],
}

# ─────────────────────────────────────────────
# 페르소나 풀 (40명)
# 다양한 연령/성별/직업/소득/특성 조합
# 실제 한국 소비자 유형을 최대한 커버하도록 설계
# ─────────────────────────────────────────────

PERSONAS = [
    {"age_group": "10대", "gender": "female", "occupation": "고등학생", "income": "용돈 월 10만원", "traits": ["SNS 활발", "트렌드 민감", "또래 의식"]},
    {"age_group": "10대", "gender": "male", "occupation": "고등학생", "income": "용돈 월 15만원", "traits": ["게임 좋아함", "가성비 중시", "유튜브 시청"]},
    {"age_group": "10대", "gender": "female", "occupation": "중학생", "income": "용돈 월 5만원", "traits": ["아이돌 팬", "틱톡 활발", "유행 민감"]},
    {"age_group": "20대", "gender": "female", "occupation": "대학생", "income": "월 150만원(알바)", "traits": ["SNS 활발", "뷰티 관심", "가성비 중시"]},
    {"age_group": "20대", "gender": "male", "occupation": "대학생", "income": "월 100만원(알바)", "traits": ["자취", "배달 자주", "가격 민감"]},
    {"age_group": "20대", "gender": "female", "occupation": "직장인(신입)", "income": "월 250만원", "traits": ["자기계발", "소확행", "인스타 활발"]},
    {"age_group": "20대", "gender": "male", "occupation": "직장인(신입)", "income": "월 280만원", "traits": ["재테크 관심", "운동", "효율 중시"]},
    {"age_group": "20대", "gender": "male", "occupation": "군인", "income": "월 100만원", "traits": ["외출 제한", "모바일 위주", "저축 중"]},
    {"age_group": "20대", "gender": "female", "occupation": "대학원생", "income": "월 120만원(조교)", "traits": ["논문 스트레스", "카페 자주", "검소"]},
    {"age_group": "30대", "gender": "female", "occupation": "직장인(과장)", "income": "월 350만원", "traits": ["워라밸", "육아", "프리미엄 선호"]},
    {"age_group": "30대", "gender": "male", "occupation": "직장인(대리)", "income": "월 320만원", "traits": ["내집마련", "실용적", "브랜드 의식"]},
    {"age_group": "30대", "gender": "female", "occupation": "프리랜서", "income": "월 300만원(변동)", "traits": ["자유로움", "건강 관심", "미니멀"]},
    {"age_group": "30대", "gender": "male", "occupation": "스타트업 대표", "income": "월 400만원(변동)", "traits": ["리스크 감수", "네트워킹", "트렌드 파악"]},
    {"age_group": "30대", "gender": "female", "occupation": "전업맘", "income": "가구소득 월 500만원", "traits": ["육아 정보", "가성비", "안전 중시"]},
    {"age_group": "40대", "gender": "male", "occupation": "직장인(부장)", "income": "월 500만원", "traits": ["가족 중심", "안정 추구", "골프"]},
    {"age_group": "40대", "gender": "female", "occupation": "자영업", "income": "월 400만원(변동)", "traits": ["실용적", "경험 중시", "교육열"]},
    {"age_group": "40대", "gender": "male", "occupation": "공무원", "income": "월 380만원", "traits": ["안정 지향", "보수적", "가족 중심"]},
    {"age_group": "40대", "gender": "female", "occupation": "간호사", "income": "월 350만원", "traits": ["건강 전문", "야간근무", "실용 중시"]},
    {"age_group": "50대", "gender": "male", "occupation": "임원", "income": "월 700만원", "traits": ["건강 관심", "보수적", "품질 중시"]},
    {"age_group": "50대", "gender": "female", "occupation": "주부", "income": "가구소득 월 600만원", "traits": ["가족 건강", "알뜰 소비", "TV 시청"]},
    {"age_group": "50대", "gender": "male", "occupation": "택시기사", "income": "월 280만원", "traits": ["장시간 근무", "실용적", "라디오 청취"]},
    {"age_group": "60대+", "gender": "male", "occupation": "은퇴", "income": "연금 월 200만원", "traits": ["건강 최우선", "보수적", "디지털 약함"]},
    {"age_group": "60대+", "gender": "female", "occupation": "은퇴", "income": "연금 월 150만원", "traits": ["손주", "건강식", "전통 선호"]},
    {"age_group": "60대+", "gender": "male", "occupation": "자영업(은퇴예정)", "income": "월 250만원", "traits": ["노후 준비", "건강검진", "등산"]},
    {"age_group": "10대", "gender": "male", "occupation": "고등학생(파트타임 스트리머)", "income": "용돈+후원 월 30만원", "traits": ["게임 방송", "장비 관심", "트위치/유튜브 활동"]},
    {"age_group": "20대", "gender": "female", "occupation": "간호사", "income": "월 300만원", "traits": ["3교대 근무", "피부관리 관심", "스트레스 해소 소비"]},
    {"age_group": "20대", "gender": "male", "occupation": "배달라이더", "income": "월 280만원(변동)", "traits": ["체력 소모 큼", "오토바이", "시간 자유"]},
    {"age_group": "30대", "gender": "female", "occupation": "IT 회사 워킹맘", "income": "월 450만원", "traits": ["육아+커리어 병행", "시간 부족", "효율 극대화"]},
    {"age_group": "30대", "gender": "male", "occupation": "요리사", "income": "월 280만원", "traits": ["식재료 민감", "야간근무 많음", "맛집 탐방"]},
    {"age_group": "30대", "gender": "female", "occupation": "초등교사", "income": "월 320만원", "traits": ["교육 관심", "방학 활용", "안정 지향"]},
    {"age_group": "40대", "gender": "male", "occupation": "택시기사(전직 사무직)", "income": "월 300만원", "traits": ["이직 경험", "장시간 운전", "팟캐스트 청취"]},
    {"age_group": "40대", "gender": "female", "occupation": "요가 강사", "income": "월 250만원(변동)", "traits": ["건강 라이프", "SNS 홍보", "소규모 클래스 운영"]},
    {"age_group": "40대", "gender": "male", "occupation": "소규모 식당 사장", "income": "월 350만원(변동)", "traits": ["배달앱 의존", "재료비 부담", "새벽 출근"]},
    {"age_group": "50대", "gender": "female", "occupation": "공인중개사", "income": "월 400만원(변동)", "traits": ["부동산 전문", "인맥 넓음", "차량 이동 많음"]},
    {"age_group": "50대", "gender": "male", "occupation": "건설 현장 근로자", "income": "월 320만원", "traits": ["체력 노동", "실용 중시", "절약형"]},
    {"age_group": "50대", "gender": "female", "occupation": "백화점 판매직", "income": "월 260만원", "traits": ["패션 트렌드 파악", "고객 응대", "서비스 마인드"]},
    {"age_group": "60대+", "gender": "female", "occupation": "은퇴(스마트폰 최근 학습)", "income": "연금 월 130만원", "traits": ["디지털 초보", "카카오톡 위주", "자녀에게 의존"]},
    {"age_group": "60대+", "gender": "male", "occupation": "은퇴(아침 운동 동호회)", "income": "연금 월 180만원", "traits": ["건강 관리", "동네 커뮤니티 활발", "TV 뉴스 시청"]},
    {"age_group": "20대", "gender": "non-binary", "occupation": "프리랜서 디자이너", "income": "월 250만원(변동)", "traits": ["젠더플루이드", "창의적", "다양성 가치관"]},
    {"age_group": "30대", "gender": "male", "occupation": "싱글대디(직장인)", "income": "월 380만원", "traits": ["혼자 육아", "시간 관리", "가성비 중시"]},
]

# ─────────────────────────────────────────────
# 평가축 매핑 (카테고리별로 다른 평가 기준 사용)
# C#의 Dictionary<string, string[]> 같은 구조
# ─────────────────────────────────────────────

AXES_MAP = {
    "식품": ["호감도", "구매의향", "가격적절성", "추천의향", "차별성"],
    "화장품": ["호감도", "구매의향", "가격적절성", "성분신뢰도", "재구매의향"],
    "앱/서비스": ["사용의향", "편의성", "필요성", "디자인호감도", "추천의향"],
    "광고": ["주목도", "메시지전달력", "브랜드연상", "호감도", "클릭의향"],
    "보험/금융": ["안전성", "보장범위", "가격적절성", "신뢰도", "가입의향"],
    "콘텐츠": ["흥미도", "몰입도", "공감도", "공유의향", "재소비의향"],
    "사업계획서": ["시장성", "실현가능성", "차별성", "수익성", "리스크"],
    "이벤트/프로모션": ["참여의향", "매력도", "혜택적절성", "공유의향", "재참여의향"],
    "패션": ["디자인호감도", "구매의향", "가격적절성", "트렌드부합", "착용의향"],
    "가전/전자": ["기능매력도", "가격적절성", "구매의향", "브랜드신뢰", "추천의향"],
    "교육": ["학습효과", "흥미도", "난이도적절성", "가격적절성", "추천의향"],
    "정책": ["필요성", "실효성", "공정성", "이해도", "지지도"],
    "부동산": ["입지매력도", "가격적절성", "투자가치", "거주의향", "추천의향"],
    "자동차": ["디자인호감도", "성능기대", "가격적절성", "구매의향", "브랜드신뢰"],
    "설문지": ["응답용이성", "질문명확성", "주제관심도", "완료의향", "소요시간적절성"],
    "기타": ["호감도", "관심도", "필요성", "추천의향", "차별성"],
}


# ─────────────────────────────────────────────
# Claude API로 학습 데이터 생성
# ─────────────────────────────────────────────

def build_generation_prompt(stimulus: str, category: str, persona: dict, axes: list[str]) -> str:
    """
    Claude에게 보낼 프롬프트 조립

    C#의 string.Format() 또는 $"" 문자열 보간과 동일.
    페르소나 정보 + 자극물 + 평가축을 하나의 프롬프트로 합침.
    """
    # 페르소나 설명 조립
    persona_desc = (
        f"- 연령대: {persona['age_group']}\n"
        f"- 성별: {'여성' if persona['gender'] == 'female' else '논바이너리' if persona['gender'] == 'non-binary' else '남성'}\n"
        f"- 직업: {persona['occupation']}\n"
        f"- 소득: {persona['income']}\n"
        f"- 특성: {', '.join(persona['traits'])}"
    )

    # JSON 출력 형식 예시 (Claude가 이 형식으로 답하도록 유도)
    axes_json_example = ", ".join([f'{{"name": "{a}", "score": 75, "reasoning": "이유"}}' for a in axes[:2]])

    return f"""당신은 한국 소비자 조사 시뮬레이션 엔진입니다.
아래 페르소나가 주어진 자극물을 접했을 때의 반응을 예측하세요.

## 페르소나
{persona_desc}

## 자극물 (카테고리: {category})
{stimulus}

## 평가 축
{', '.join(axes)}

## 출력 형식 (반드시 이 JSON만 출력)
{{"evaluations": [{axes_json_example}, ...], "open_response": "한 줄 반응", "confidence": 0.75}}

## 규칙
- 각 축을 0-100으로 평가 (50이 중립)
- reasoning은 페르소나의 말투와 관점으로 1~2문장
- open_response는 이 사람이 실제로 할 법한 자연스러운 한마디
- 점수는 페르소나 특성에 맞게 현실적으로 (60대는 앱 점수 낮고, 10대는 보험 관심 낮음)
- confidence는 0.5~0.9 사이
- JSON 외 다른 텍스트 절대 없이"""


def parse_json_response(text: str) -> dict | None:
    """
    Claude 응답에서 JSON 추출 (다양한 형식 대응)

    Claude가 가끔 ```json ... ``` 형태로 감싸거나 앞뒤에 텍스트를 붙이므로
    여러 방법으로 JSON 추출을 시도함.

    C#의 JsonSerializer.Deserialize() + 전처리 같은 역할
    """
    text = text.strip()

    # 방법 1: 마크다운 코드블록 (```json ... ```) 제거
    if "```" in text:
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # 방법 2: { } 로 감싸진 JSON 객체 추출
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # 방법 3: 텍스트 전체를 직접 JSON 파싱 시도
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


async def generate_one(
    client: anthropic.AsyncAnthropic,
    stimulus: str,
    category: str,
    persona: dict,
    axes: list[str],
    model: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """
    학습 데이터 1건 생성 (동시성 제한 적용)

    semaphore = C#의 SemaphoreSlim 과 동일한 역할.
    동시에 너무 많은 API 호출을 하면 Rate Limit에 걸리므로
    semaphore로 동시 실행 수를 제한함.
    """
    # semaphore로 동시 실행 수 제한 (C#의 await semaphore.WaitAsync())
    async with semaphore:
        prompt = build_generation_prompt(stimulus, category, persona, axes)

        try:
            # Claude API 호출
            response = await client.messages.create(
                model=model,
                max_tokens=1024,            # 최대 응답 길이
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            output = parse_json_response(text)  # JSON 파싱

            # 파싱 실패 체크
            if output is None:
                print(f"  JSON 파싱 실패")
                return None

            # 기본 검증: evaluations 필드가 있고 리스트인지
            if "evaluations" not in output or not isinstance(output["evaluations"], list):
                print(f"  evaluations 필드 누락")
                return None

            # 학습 데이터 형식으로 조립 (input + output 쌍)
            return {
                "input": {
                    "stimulus": stimulus,
                    "stimulus_type": category,
                    "persona_age_group": persona["age_group"],
                    "persona_gender": persona["gender"],
                    "persona_occupation": persona["occupation"],
                    "persona_income": persona["income"],
                    "persona_traits": persona["traits"],
                    "axes": axes,
                },
                "output": output,
            }
        except anthropic.RateLimitError:
            # Rate Limit 걸리면 30초 대기 후 재시도 (None 반환)
            print(f"  Rate limited, waiting 30s...")
            await asyncio.sleep(30)
            return None
        except Exception as e:
            print(f"  Error: {e}")
            return None


def validate_output_schema(output: dict) -> bool:
    """
    Validate that generated output matches required schema:
    - Must have 'evaluations' array
    - Each evaluation must have 'score' (numeric) and 'reasoning' (string)
    """
    evals = output.get("evaluations")
    if not isinstance(evals, list) or len(evals) == 0:
        return False
    for ev in evals:
        if not isinstance(ev, dict):
            return False
        if "score" not in ev or "reasoning" not in ev:
            return False
        if not isinstance(ev["score"], (int, float)):
            return False
        if not isinstance(ev["reasoning"], str) or not ev["reasoning"].strip():
            return False
    return True


def load_existing_keys(output_path: str) -> set[str]:
    """
    Load existing stimulus+persona dedup keys from the output file.
    Key = stimulus text + persona age_group + persona gender + persona occupation
    """
    keys = set()
    path = Path(output_path)
    if not path.exists():
        return keys
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                inp = item.get("input", {})
                key = f"{inp.get('stimulus', '')}|{inp.get('persona_age_group', '')}|{inp.get('persona_gender', '')}|{inp.get('persona_occupation', '')}"
                keys.add(key)
            except json.JSONDecodeError:
                continue
    return keys


def make_dedup_key(stimulus: str, persona: dict) -> str:
    return f"{stimulus}|{persona['age_group']}|{persona['gender']}|{persona['occupation']}"


async def generate_batch(count: int, output_path: str, model: str, concurrent: int):
    """
    학습 데이터 배치 생성 (메인 생성 루프)
    C#의 Parallel.ForEachAsync() 같은 병렬 처리.

    count건의 데이터를 concurrent개씩 동시에 생성.
    결과는 JSONL 파일에 한 건씩 즉시 저장 (중간에 끊겨도 데이터 보존).
    """
    # 비동기 Anthropic 클라이언트 생성 (환경변수에서 API 키 자동 로드)
    client = anthropic.AsyncAnthropic()
    # 동시 실행 제한 (C#의 new SemaphoreSlim(concurrent))
    semaphore = asyncio.Semaphore(concurrent)

    categories = list(STIMULI.keys())
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing entries for deduplication
    existing_keys = load_existing_keys(output_path)
    print(f"Loaded {len(existing_keys)} existing entries for dedup check")

    print(f"Generating {count} training examples...")
    print(f"Model: {model}")
    print(f"Concurrent: {concurrent}")
    print(f"Output: {output_path}")
    print(f"Categories: {len(categories)}, Stimuli: {sum(len(v) for v in STIMULI.values())}, Personas: {len(PERSONAS)}")
    print()

    generated = 0       # 성공 건수
    errors = 0           # 실패 건수
    skipped_dup = 0      # 중복 스킵 건수
    skipped_schema = 0   # 스키마 검증 실패 건수
    batch_size = min(concurrent * 2, count)  # 한 배치에 처리할 건수
    consecutive_rate_limits = 0  # Rate limit 연속 횟수

    # "a" = append 모드 (이어쓰기) → 중간에 중단해도 이전 데이터 보존
    with open(output_file, "a", encoding="utf-8") as f:
        for batch_start in range(0, count, batch_size):
            batch_end = min(batch_start + batch_size, count)
            tasks = []

            # 배치 내 각 건에 대해 랜덤으로 카테고리/자극물/페르소나 조합 선택
            for i in range(batch_start, batch_end):
                category = random.choice(categories)            # 랜덤 카테고리
                stimulus = random.choice(STIMULI[category])     # 랜덤 자극물
                persona = random.choice(PERSONAS)               # 랜덤 페르소나
                axes = AXES_MAP[category]                       # 카테고리에 맞는 평가축

                # 20% 확률로 평가축 일부만 사용 (다양성 확보)
                if random.random() < 0.2:
                    axes = random.sample(axes, k=random.randint(3, len(axes)))

                tasks.append((
                    generate_one(client, stimulus, category, persona, axes, model, semaphore),
                    f"[{i+1}/{count}] {category} | {persona['age_group']} {persona['gender']} {persona['occupation']}",
                    make_dedup_key(stimulus, persona),
                ))

            # 배치 내 모든 태스크를 동시 실행 (C#의 Task.WhenAll() 같은 것)
            results = await asyncio.gather(*[t[0] for t in tasks], return_exceptions=True)

            # 결과 처리 + 파일 저장
            batch_rate_limited = False
            for (_, desc, dedup_key), result in zip(tasks, results):
                if isinstance(result, Exception):
                    print(f"{desc} → ERROR: {result}")
                    errors += 1
                elif result is None:
                    print(f"{desc} → FAILED")
                    errors += 1
                else:
                    # Dedup check
                    if dedup_key in existing_keys:
                        print(f"{desc} → SKIPPED (duplicate)")
                        skipped_dup += 1
                        continue

                    # Schema validation
                    if not validate_output_schema(result.get("output", {})):
                        print(f"{desc} → SKIPPED (schema invalid: missing score/reasoning in evaluations)")
                        skipped_schema += 1
                        continue

                    # 성공 → JSONL에 한 줄로 저장
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()  # 즉시 디스크에 쓰기 (버퍼링 방지)
                    existing_keys.add(dedup_key)
                    generated += 1
                    consecutive_rate_limits = 0
                    print(f"{desc} → OK")

            print(f"  Progress: {generated}/{count} generated, {errors} errors, {skipped_dup} dups, {skipped_schema} schema fails")

            # Rate limit backoff: if entire batch failed, wait progressively longer
            if generated == 0 and errors == batch_end - batch_start:
                consecutive_rate_limits += 1
                wait_time = min(30 * consecutive_rate_limits, 120)
                print(f"  All requests in batch failed. Saving partial results and waiting {wait_time}s...")
                f.flush()
                await asyncio.sleep(wait_time)

    print(f"\nDone! Generated {generated} examples ({errors} errors, {skipped_dup} dups, {skipped_schema} schema fails)")
    print(f"Saved to {output_path}")


def main():
    """
    CLI 진입점 (C#의 static void Main(string[] args))

    커맨드라인 인자:
    --count 3000         : 생성할 데이터 수
    --model claude-...   : 사용할 Claude 모델
    --concurrent 10      : 동시 요청 수
    --output path.jsonl  : 출력 파일 경로
    """
    # argparse = C#의 CommandLineParser 라이브러리 같은 것
    parser = argparse.ArgumentParser(description="AIPA-Eval 학습 데이터 생성")
    parser.add_argument("--count", type=int, default=100, help="생성할 데이터 수")
    parser.add_argument(
        "--output", type=str,
        default=f"data/training/eval_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
        help="출력 파일 경로",
    )
    parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001", help="Claude 모델")
    parser.add_argument("--concurrent", type=int, default=5, help="동시 요청 수")
    args = parser.parse_args()

    # asyncio.run() = 비동기 함수를 동기적으로 실행 (C#의 .GetAwaiter().GetResult() 같은 것)
    asyncio.run(generate_batch(args.count, args.output, args.model, args.concurrent))


if __name__ == "__main__":
    main()
