"""
AIPA 페르소나 임베딩 모델 학습

자극물 평가 데이터(JSONL)에서 패턴을 추출해서
페르소나 속성 + 자극물 카테고리 → 점수를 예측하는 경량 모델.

LLM 없이 0.01초 만에 점수 예측 가능.
CPU에서 동작, 모델 크기 ~1MB.

사용법:
    python training/train_embedding.py
    python training/train_embedding.py --data training/data/training_data_deduped.jsonl --epochs 100
"""

import json
import random
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ========== 1. 데이터 전처리 ==========

# 카테고리 인코딩용 사전 (문자열 → 숫자 ID)
AGE_GROUPS = ["10대", "20대", "30대", "40대", "50대", "60대+"]
GENDERS = ["male", "female", "non-binary"]
OCCUPATIONS = [
    "고등학생", "중학생", "대학생", "대학원생", "군인",
    "직장인(신입)", "직장인(대리)", "직장인(과장)", "직장인(부장)",
    "프리랜서", "프리랜서 디자이너", "자영업", "스타트업 대표",
    "전업맘", "주부", "공무원", "임원", "택시기사",
    "은퇴", "자영업(은퇴예정)",
    "간호사", "배달 라이더", "헬스 트레이너", "웹디자이너",
    "요리사", "초등교사", "싱글대디(직장인)", "IT 워킹맘",
    "부동산 중개사", "요가 강사", "소방관", "쇼핑몰 운영자",
    "소규모 식당 사장", "학원 원장", "택배 기사", "백화점 직원",
    "건설 현장 소장", "백화점 판매원", "버스 기사", "자동차 정비사",
    "경비원", "문화센터 수강생", "파트타임 스트리머",
    # fallback
    "기타",
]
TRAITS = [
    "SNS 활발", "트렌드 민감", "또래 의식", "게임 좋아함", "가성비 중시",
    "유튜브 시청", "아이돌 팬", "틱톡 활발", "유행 민감", "뷰티 관심",
    "자취", "배달 자주", "가격 민감", "자기계발", "소확행", "인스타 활발",
    "재테크 관심", "운동", "효율 중시", "외출 제한", "모바일 위주",
    "저축 중", "논문 스트레스", "카페 자주", "검소", "워라밸", "육아",
    "프리미엄 선호", "내집마련", "실용적", "브랜드 의식", "자유로움",
    "건강 관심", "미니멀", "리스크 감수", "네트워킹", "트렌드 파악",
    "육아 정보", "안전 중시", "가족 중심", "안정 추구", "골프",
    "경험 중시", "교육열", "안정 지향", "보수적", "건강 전문",
    "야간근무", "실용 중시", "품질 중시", "알뜰 소비", "TV 시청",
    "장시간 근무", "라디오 청취", "건강 최우선", "디지털 약함",
    "손주", "건강식", "전통 선호", "노후 준비", "건강검진", "등산",
    "콘텐츠 제작", "트렌드 선도", "장비 투자", "야간 생활",
    "가격 극도로 민감", "에너지드링크", "3교대", "스트레스 관리",
    "체력 소모", "시간 자유", "오토바이", "화장품 전문", "리뷰 활발",
    "운동 전문", "식단 관리", "보충제", "디자인 감각", "맥북 유저",
    "카페 작업", "식재료 민감", "야근 많음", "맛집 탐방", "안정적",
    "교육 관심", "방학 활용", "시간 부족", "재택근무", "육아+일 병행",
    "부동산 전문", "자차 필수", "비건", "명상", "체력 관리", "교대근무",
    "마케팅", "SNS 필수", "자영업 스트레스", "배달앱 의존", "가족 부양",
    "웰빙", "규율적", "적응 중", "학부모 소통", "경영", "새벽 출근",
    "패션 관심", "고객 응대", "브랜드 지식", "인맥 넓음", "카톡 활발",
    "현장 관리", "규칙적", "가족 건강 관리", "기술직", "차량 전문",
    "디지털 초보", "손주 사진", "등산 동호회", "절약", "배움 욕구",
    "사교적", "취미 활동", "근검절약",
    # fallback
    "기타",
]
CATEGORIES = [
    "식품", "화장품", "앱/서비스", "광고", "보험/금융", "콘텐츠",
    "사업계획서", "이벤트/프로모션", "패션", "가전/전자",
    "교육", "정책", "부동산", "자동차", "설문지", "기타",
]
AXES = [
    "호감도", "구매의향", "가격적절성", "추천의향", "차별성",
    "성분신뢰도", "재구매의향", "사용의향", "편의성", "필요성",
    "디자인호감도", "주목도", "메시지전달력", "브랜드연상", "클릭의향",
    "안전성", "보장범위", "신뢰도", "가입의향", "흥미도", "몰입도",
    "공감도", "공유의향", "재소비의향", "시장성", "실현가능성",
    "수익성", "리스크", "참여의향", "매력도", "혜택적절성", "재참여의향",
    "트렌드부합", "착용의향", "기능매력도", "브랜드신뢰", "성능기대",
    "학습효과", "난이도적절성", "실효성", "공정성", "이해도", "지지도",
    "입지매력도", "투자가치", "거주의향",
    "응답용이성", "질문명확성", "주제관심도", "완료의향", "소요시간적절성",
    "관심도",
]


def safe_index(vocab: list, value: str, fallback_idx: int = -1) -> int:
    """vocab에서 value 찾기. 없으면 마지막 인덱스(기타) 반환"""
    try:
        return vocab.index(value)
    except ValueError:
        return len(vocab) - 1  # "기타" 인덱스


def encode_traits(persona_traits: list[str], max_traits: int = 5) -> list[int]:
    """특성 리스트를 고정 길이 인덱스 배열로 변환"""
    indices = []
    for t in persona_traits[:max_traits]:
        indices.append(safe_index(TRAITS, t))
    # 패딩 (부족하면 0으로 채움)
    while len(indices) < max_traits:
        indices.append(0)
    return indices


class EvalDataset(Dataset):
    """학습 데이터셋: JSONL → (페르소나 특성, 카테고리, 축) → 점수"""

    def __init__(self, data_path: str):
        self.samples = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                inp = item.get("input", {})
                out = item.get("output", {})
                evaluations = out.get("evaluations", [])

                if not evaluations:
                    continue

                # 페르소나 특성 인코딩
                age_idx = safe_index(AGE_GROUPS, inp.get("persona_age_group", ""))
                gender_idx = safe_index(GENDERS, inp.get("persona_gender", ""))
                occ_idx = safe_index(OCCUPATIONS, inp.get("persona_occupation", ""))
                trait_indices = encode_traits(inp.get("persona_traits", []))
                cat_idx = safe_index(CATEGORIES, inp.get("stimulus_type", ""))

                # 각 평가축별로 하나의 샘플 생성
                for ev in evaluations:
                    axis_idx = safe_index(AXES, ev.get("name", ""))
                    score = ev.get("score", 50)
                    if isinstance(score, (int, float)):
                        score = max(0, min(100, score))
                    else:
                        continue

                    self.samples.append({
                        "age": age_idx,
                        "gender": gender_idx,
                        "occupation": occ_idx,
                        "traits": trait_indices,
                        "category": cat_idx,
                        "axis": axis_idx,
                        "score": score / 100.0,  # 0~1로 정규화
                    })

        print(f"  로드 완료: {len(self.samples)}건 (평가축 단위)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "age": torch.tensor(s["age"], dtype=torch.long),
            "gender": torch.tensor(s["gender"], dtype=torch.long),
            "occupation": torch.tensor(s["occupation"], dtype=torch.long),
            "traits": torch.tensor(s["traits"], dtype=torch.long),
            "category": torch.tensor(s["category"], dtype=torch.long),
            "axis": torch.tensor(s["axis"], dtype=torch.long),
            "score": torch.tensor(s["score"], dtype=torch.float32),
        }


# ========== 2. 모델 정의 ==========

class PersonaEmbeddingModel(nn.Module):
    """
    페르소나 임베딩 모델

    페르소나 속성(연령, 성별, 직업, 특성)과
    평가 컨텍스트(카테고리, 평가축)를 벡터로 변환하고,
    이들의 상호작용으로 점수를 예측.

    파라미터 수: ~50,000 (Qwen 7B의 1/140,000)
    """

    def __init__(
        self,
        embed_dim: int = 24,
        hidden_dim: int = 64,
        n_ages: int = len(AGE_GROUPS),
        n_genders: int = len(GENDERS),
        n_occupations: int = len(OCCUPATIONS),
        n_traits: int = len(TRAITS),
        n_categories: int = len(CATEGORIES),
        n_axes: int = len(AXES),
        max_traits: int = 5,
    ):
        super().__init__()

        # 각 속성을 벡터로 변환하는 임베딩 레이어
        self.age_embed = nn.Embedding(n_ages, embed_dim)
        self.gender_embed = nn.Embedding(n_genders, embed_dim // 2)
        self.occupation_embed = nn.Embedding(n_occupations, embed_dim)
        self.trait_embed = nn.Embedding(n_traits, embed_dim)
        self.category_embed = nn.Embedding(n_categories, embed_dim)
        self.axis_embed = nn.Embedding(n_axes, embed_dim)

        # 특성 집계 (여러 특성을 하나로 합침)
        self.trait_attention = nn.Linear(embed_dim, 1)

        # 페르소나 벡터 생성
        persona_input_dim = embed_dim + (embed_dim // 2) + embed_dim + embed_dim  # age + gender + occ + traits
        self.persona_encoder = nn.Sequential(
            nn.Linear(persona_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embed_dim),
        )

        # 컨텍스트 벡터 생성 (카테고리 + 평가축)
        context_input_dim = embed_dim + embed_dim  # category + axis
        self.context_encoder = nn.Sequential(
            nn.Linear(context_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # 최종 점수 예측
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # 0~1 출력
        )

    def forward(self, age, gender, occupation, traits, category, axis):
        # 각 속성을 벡터로 변환
        age_vec = self.age_embed(age)                    # (batch, embed_dim)
        gender_vec = self.gender_embed(gender)           # (batch, embed_dim//2)
        occ_vec = self.occupation_embed(occupation)      # (batch, embed_dim)

        # 특성 벡터 (여러 개를 attention으로 합침)
        trait_vecs = self.trait_embed(traits)             # (batch, 5, embed_dim)
        trait_weights = torch.softmax(self.trait_attention(trait_vecs), dim=1)  # (batch, 5, 1)
        trait_vec = (trait_vecs * trait_weights).sum(dim=1)  # (batch, embed_dim)

        # 페르소나 벡터 조합
        persona_input = torch.cat([age_vec, gender_vec, occ_vec, trait_vec], dim=-1)
        persona_vec = self.persona_encoder(persona_input)  # (batch, embed_dim)

        # 컨텍스트 벡터
        cat_vec = self.category_embed(category)
        axis_vec = self.axis_embed(axis)
        context_input = torch.cat([cat_vec, axis_vec], dim=-1)
        context_vec = self.context_encoder(context_input)  # (batch, embed_dim)

        # 페르소나 × 컨텍스트 → 점수 예측
        combined = torch.cat([persona_vec, context_vec], dim=-1)
        score = self.predictor(combined).squeeze(-1)       # (batch,)

        return score


# ========== 3. 학습 ==========

def train_model(args):
    print("=" * 50)
    print("AIPA 페르소나 임베딩 모델 학습")
    print("=" * 50)

    # 데이터 로드
    print(f"\n[1/4] 데이터 로드: {args.data}")
    dataset = EvalDataset(args.data)

    if len(dataset) < 100:
        print("  데이터가 100건 미만입니다. 더 많은 데이터가 필요합니다.")
        return

    # Train/Eval 분할
    train_size = int(len(dataset) * 0.9)
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"  학습: {train_size}건, 검증: {eval_size}건")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)

    # 모델 생성
    print(f"\n[2/4] 모델 생성")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PersonaEmbeddingModel(embed_dim=args.embed_dim, hidden_dim=args.hidden_dim).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  파라미터 수: {total_params:,} ({total_params/1000:.1f}K)")
    print(f"  디바이스: {device}")

    # 손실함수 + 옵티마이저
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 학습 로그
    log_path = Path(args.output) / "embedding_training_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 학습 루프
    print(f"\n[3/4] 학습 시작 (epochs: {args.epochs})")
    best_eval_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0
        train_count = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            pred = model(
                batch["age"], batch["gender"], batch["occupation"],
                batch["traits"], batch["category"], batch["axis"],
            )
            loss = criterion(pred, batch["score"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(pred)
            train_count += len(pred)

        scheduler.step()
        avg_train_loss = train_loss / train_count

        # ── Eval ──
        model.eval()
        eval_loss = 0
        eval_count = 0
        eval_errors = []

        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                pred = model(
                    batch["age"], batch["gender"], batch["occupation"],
                    batch["traits"], batch["category"], batch["axis"],
                )
                loss = criterion(pred, batch["score"])
                eval_loss += loss.item() * len(pred)
                eval_count += len(pred)

                # 점수 오차 (0~100 스케일)
                errors = torch.abs(pred - batch["score"]) * 100
                eval_errors.extend(errors.cpu().tolist())

        avg_eval_loss = eval_loss / eval_count
        avg_error = np.mean(eval_errors)
        median_error = np.median(eval_errors)

        # 로그 저장
        log_entry = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "eval_loss": round(avg_eval_loss, 6),
            "avg_error": round(avg_error, 2),
            "median_error": round(median_error, 2),
            "lr": round(scheduler.get_last_lr()[0], 8),
            "timestamp": datetime.now().isoformat(),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        # 10 에포크마다 출력
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"train_loss: {avg_train_loss:.4f} | "
                  f"eval_loss: {avg_eval_loss:.4f} | "
                  f"avg_error: {avg_error:.1f}점 | "
                  f"median_error: {median_error:.1f}점")

        # Best 모델 저장
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            save_path = Path(args.output) / "persona_embedding_model.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "embed_dim": args.embed_dim,
                    "hidden_dim": args.hidden_dim,
                    "total_params": total_params,
                    "best_eval_loss": best_eval_loss,
                    "best_avg_error": avg_error,
                    "best_median_error": median_error,
                    "train_size": train_size,
                    "eval_size": eval_size,
                    "epochs_trained": epoch,
                },
                "vocab": {
                    "age_groups": AGE_GROUPS,
                    "genders": GENDERS,
                    "occupations": OCCUPATIONS,
                    "traits": TRAITS,
                    "categories": CATEGORIES,
                    "axes": AXES,
                },
            }, save_path)

    # 완료
    print(f"\n[4/4] 학습 완료!")
    print(f"  Best eval loss: {best_eval_loss:.4f}")
    save_path = Path(args.output) / "persona_embedding_model.pt"
    size_mb = save_path.stat().st_size / 1024 / 1024
    print(f"  모델 저장: {save_path} ({size_mb:.2f}MB)")
    print(f"  로그 저장: {log_path}")


# ========== 4. 추론 테스트 ==========

def test_inference(model_path: str):
    """학습된 모델로 추론 테스트"""
    print("\n" + "=" * 50)
    print("추론 테스트")
    print("=" * 50)

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    vocab = checkpoint["vocab"]

    model = PersonaEmbeddingModel(
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"  모델 로드 완료 (파라미터: {config['total_params']:,})")
    print(f"  학습 데이터: {config['train_size']}건")
    print(f"  평균 오차: {config['best_avg_error']:.1f}점")

    # 테스트 케이스
    test_cases = [
        ("20대", "female", "대학생", ["가성비 중시", "SNS 활발"], "식품", "구매의향"),
        ("20대", "female", "대학생", ["가성비 중시", "SNS 활발"], "가전/전자", "구매의향"),
        ("50대", "male", "임원", ["품질 중시", "건강 관심"], "식품", "구매의향"),
        ("50대", "male", "임원", ["품질 중시", "건강 관심"], "가전/전자", "구매의향"),
        ("10대", "female", "고등학생", ["SNS 활발", "트렌드 민감"], "패션", "호감도"),
        ("60대+", "male", "은퇴", ["건강 최우선", "보수적"], "앱/서비스", "사용의향"),
    ]

    print("\n  [테스트 결과]")
    for age, gender, occ, traits, cat, axis in test_cases:
        age_idx = torch.tensor([safe_index(AGE_GROUPS, age)])
        gender_idx = torch.tensor([safe_index(GENDERS, gender)])
        occ_idx = torch.tensor([safe_index(OCCUPATIONS, occ)])
        trait_idx = torch.tensor([encode_traits(traits)])
        cat_idx = torch.tensor([safe_index(CATEGORIES, cat)])
        axis_idx = torch.tensor([safe_index(AXES, axis)])

        with torch.no_grad():
            score = model(age_idx, gender_idx, occ_idx, trait_idx, cat_idx, axis_idx)
            score_100 = int(score.item() * 100)

        trait_str = "+".join(traits[:2])
        print(f"  {age} {gender} {occ}({trait_str}) → {cat}/{axis}: {score_100}점")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIPA 페르소나 임베딩 모델 학습")
    parser.add_argument("--data", type=str, default="training/data/training_data_deduped.jsonl")
    parser.add_argument("--output", type=str, default="training/models/embedding")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=24)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    train_model(args)

    # 학습 후 추론 테스트
    model_path = Path(args.output) / "persona_embedding_model.pt"
    if model_path.exists():
        test_inference(str(model_path))
