"""
Direction Accuracy 측정 — Max/Min 트릭 BEFORE/AFTER 동시 비교
- 학습된 V5 모델을 그대로 로드해서 동일 split의 test set으로 평가
- BEFORE: mean compression DA (3-step 평균을 단일 신호로 압축)
- AFTER : Entry+Exit DA 평균 (max/min 압축)
- 5종목 종목별 + 평균 표 출력
"""
import os
import sys
import json
import pickle
import random

# Windows cp949 stdout에서 유니코드 문자(엠대시 등) 출력 가능하도록 UTF-8 강제
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import torch
import FinanceDataReader as fdr

# 시드 고정 (재현성)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from lstm_train_v5 import (
    StockLSTM_V5,
    create_features_v4,
    get_external_data,
    generate_foundation_predictions,
)

CODES = ['KS11', 'KQ11', '005930', '000660', '035720']
WINDOW = 60
START_DATE = '2015-01-01'
MODEL_DIR = os.path.join(SCRIPT_DIR, 'models_v5')

# Confidence Gating 하이퍼파라미터
# - ENTRY_THR / EXIT_THR : 조원분 ETH 최적값 그대로 사용
# - CONF_MULT            : ETH(20) → 한국 주식 변동성(~1/10) 보정 위해 200으로 스케일업
CONF_MULT = 200.0
ENTRY_THR = 0.21
EXIT_THR = 0.32

# MODIFIED [Multi-Filter]: 다중 필터 임계값 (ETH-튜닝값을 우리 BB 공식으로 변환)
# 우리 bb_position = (close - lower) / (upper - lower) ∈ [0, 1] (mid=0.5)
# ETH bb_position  = (close - mid)   / (upper - mid)   ∈ [-1, 1] (mid=0)
# 변환식: our_bb = (eth_bb + 1) / 2
# - eth -0.43 → 0.285 (하단 근처)
# - eth  0.02 → 0.51  (중간 살짝 위)
# - eth  1.16 → 1.08  (상단 돌파)
# - eth -0.77 → 0.115 (하단 이탈)
RSI_ENTRY_LOW = 38.0    # 반등 패턴: rsi < 38 (과매도)
BB_ENTRY_LOW = 0.285    # 반등 패턴: bb_pos < 0.285 (하단 터치)
RSI_ENTRY_HIGH = 31.0   # 돌파 패턴: rsi > 31
BB_ENTRY_HIGH = 0.51    # 돌파 패턴: bb_pos > 0.51 (중간 위)
RSI_EXIT_HIGH = 65.0    # 과열 패턴: rsi > 65 (과매수)
BB_EXIT_HIGH = 1.08     # 과열 패턴: bb_pos > 1.08 (상단 돌파)
BB_EXIT_LOW = 0.115     # 붕괴 패턴: bb_pos < 0.115 (하단 이탈)


def measure_one(code: str) -> dict:
    safe_code = code.replace('^', '').replace('/', '_')

    # 1. 데이터 로드
    df = fdr.DataReader(code, START_DATE)
    if df.empty or len(df) < 200:
        raise RuntimeError(f"{code}: 데이터 부족 ({len(df)}일)")

    close_mean = df['Close'].mean()

    # 2. Foundation Model 예측 (캐시 hit 기대)
    foundation_preds = generate_foundation_predictions(df, code, close_mean, MODEL_DIR)

    # 3. 외부 지표
    external_data = get_external_data(START_DATE, df.index[-1].strftime('%Y-%m-%d'))

    # 4. 특성
    features = create_features_v4(df, external_data, foundation_preds)

    # 5. 저장된 scaler/meta 로드
    with open(os.path.join(MODEL_DIR, f'{safe_code}_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, f'{safe_code}_target_scaler.pkl'), 'rb') as f:
        target_scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, f'{safe_code}_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)

    # 컬럼 순서 맞추기 (학습 시와 동일)
    expected_cols = meta['feature_columns']
    for col in expected_cols:
        if col not in features.columns:
            features[col] = 0
    features = features[expected_cols]

    scaled = scaler.transform(features.values)
    close_raw = df['Close'].values

    # 6. 윈도우 + 타겟 (학습 시와 동일 로직)
    X, y = [], []
    for i in range(WINDOW, len(scaled) - 3):
        X.append(scaled[i - WINDOW:i])
        p_now = close_raw[i]
        y.append([
            (close_raw[i + 1] - p_now) / p_now,
            (close_raw[i + 2] - p_now) / p_now,
            (close_raw[i + 3] - p_now) / p_now,
        ])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # 7. 80:20 시간순 split (학습 시와 동일)
    split = int(len(X) * 0.8)
    X_train = X[:split]
    y_train_raw = y[:split]
    X_test = X[split:]
    y_test_raw = y[split:]

    # 8. 모델 로드 + 예측
    input_size = X.shape[-1]
    model = StockLSTM_V5(input_size=input_size)
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, f'{safe_code}_lstm_v5.pth'),
        map_location='cpu', weights_only=True
    ))
    model.eval()

    with torch.no_grad():
        pred_scaled = model(torch.FloatTensor(X_test)).numpy()
    pred_raw_uncorrected = target_scaler.inverse_transform(pred_scaled)  # (N, 3) 수익률

    # MODIFIED [Systematic Bias]: train set으로 bias 계산 (이미 학습된 모델 그대로 사용)
    # bias = mean(y_train) - mean(pred_train), raw 수익률 단위 스칼라 1개
    with torch.no_grad():
        pred_train_scaled = model(torch.FloatTensor(X_train)).numpy()
    pred_train_raw = target_scaler.inverse_transform(pred_train_scaled)
    systematic_bias = float(y_train_raw.mean() - pred_train_raw.mean())

    # MODIFIED [Systematic Bias]: bias를 meta에 저장해서 lstm_service.py가 실시간 예측에서 사용
    meta['systematic_bias'] = systematic_bias
    # MODIFIED [Multi-Filter]: 다중 필터 임계값을 meta에 저장 (서비스 추론에서 동일 적용)
    meta['filter_rsi_entry_low'] = RSI_ENTRY_LOW
    meta['filter_bb_entry_low'] = BB_ENTRY_LOW
    meta['filter_rsi_entry_high'] = RSI_ENTRY_HIGH
    meta['filter_bb_entry_high'] = BB_ENTRY_HIGH
    meta['filter_rsi_exit_high'] = RSI_EXIT_HIGH
    meta['filter_bb_exit_high'] = BB_EXIT_HIGH
    meta['filter_bb_exit_low'] = BB_EXIT_LOW
    with open(os.path.join(MODEL_DIR, f'{safe_code}_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    # MODIFIED [Systematic Bias]: 모든 예측에 bias 더함 (모든 step 동일하게 broadcast)
    pred_raw = pred_raw_uncorrected + systematic_bias

    # 9. 메트릭
    # 기존 per-step DA (참고용)
    da_t1 = (np.sign(y_test_raw[:, 0]) == np.sign(pred_raw[:, 0])).mean() * 100
    da_t2 = (np.sign(y_test_raw[:, 1]) == np.sign(pred_raw[:, 1])).mean() * 100
    da_t3 = (np.sign(y_test_raw[:, 2]) == np.sign(pred_raw[:, 2])).mean() * 100

    # BEFORE: 평균 압축 DA
    pred_mean = pred_raw.mean(axis=1)
    actual_mean = y_test_raw.mean(axis=1)
    da_mean_comp = (np.sign(pred_mean) == np.sign(actual_mean)).mean() * 100

    # max/min 트릭 단계 메트릭 (이전 단계 결과 = 이번 단계 BEFORE)
    pred_max = pred_raw.max(axis=1)
    actual_max = y_test_raw.max(axis=1)
    da_entry = (np.sign(pred_max) == np.sign(actual_max)).mean() * 100

    pred_min = pred_raw.min(axis=1)
    actual_min = y_test_raw.min(axis=1)
    da_exit = (np.sign(pred_min) == np.sign(actual_min)).mean() * 100

    da_max_min_avg = (da_entry + da_exit) / 2  # max/min 트릭 단계 결과

    # ─────────────────────────────────────────────────────────────
    # MODIFIED [Confidence Gating]: max/min return → confidence → threshold gate
    # confidence = min(|return| * MULT, 1.0). gate 통과한 표본만 DA 계산.
    # ─────────────────────────────────────────────────────────────
    def _gated_da(pmax, pmin, amax, amin):
        """공통 계산기 — bias 적용 전/후 둘 다 같은 로직으로 측정."""
        ce = np.minimum(np.abs(pmax) * CONF_MULT, 1.0)
        ci = np.minimum(np.abs(pmin) * CONF_MULT, 1.0)
        em = ce > ENTRY_THR
        xm = ci > EXIT_THR
        ne, nx = int(em.sum()), int(xm.sum())
        de = float((np.sign(pmax[em]) == np.sign(amax[em])).mean() * 100) if ne > 0 else None
        dx = float((np.sign(pmin[xm]) == np.sign(amin[xm])).mean() * 100) if nx > 0 else None
        if de is not None and dx is not None:
            avg = (de + dx) / 2
        elif de is not None:
            avg = de
        elif dx is not None:
            avg = dx
        else:
            avg = None
        return ne, nx, de, dx, avg

    # AFTER: bias 보정된 pred_raw로 게이팅
    n_entry_gated, n_exit_gated, da_entry_gated, da_exit_gated, da_gated_avg = \
        _gated_da(pred_max, pred_min, actual_max, actual_min)

    # MODIFIED [Systematic Bias]: BEFORE — bias 보정 안 한 원본 예측으로 같은 게이팅 적용
    # → 이게 이번 단계의 진짜 BEFORE (= 이전 단계 게이팅 결과와 동일해야 함)
    pred_max_nb = pred_raw_uncorrected.max(axis=1)
    pred_min_nb = pred_raw_uncorrected.min(axis=1)
    n_entry_nb, n_exit_nb, da_entry_nb, da_exit_nb, da_gated_nb_avg = \
        _gated_da(pred_max_nb, pred_min_nb, actual_max, actual_min)

    # ─────────────────────────────────────────────────────────────
    # MODIFIED [Multi-Filter]: Confidence Gating 위에 RSI/BB/MACD 패턴 AND 추가
    # 진입: pattern_rebound(과매도+하단) OR pattern_breakout(돌파+모멘텀)
    # 청산: pattern_overheat(과매수+상단) OR pattern_breakdown(하단이탈+모멘텀-)
    # ─────────────────────────────────────────────────────────────
    # 학습 시와 동일한 인덱스 매핑: test 샘플 j의 "현재" 시점 글로벌 인덱스 = split + j + WINDOW
    test_now_idx = split + WINDOW
    rsi_vals = features['rsi'].values
    bb_vals = features['bb_position'].values
    macd_vals = features['macd_hist'].values
    rsi_test = rsi_vals[test_now_idx:test_now_idx + len(X_test)]
    bb_test = bb_vals[test_now_idx:test_now_idx + len(X_test)]
    macd_test = macd_vals[test_now_idx:test_now_idx + len(X_test)]

    pattern_rebound = (rsi_test < RSI_ENTRY_LOW) & (bb_test < BB_ENTRY_LOW)
    pattern_breakout = (rsi_test > RSI_ENTRY_HIGH) & (bb_test > BB_ENTRY_HIGH) & (macd_test > 0)
    entry_pattern = pattern_rebound | pattern_breakout

    pattern_overheat = (rsi_test > RSI_EXIT_HIGH) & (bb_test > BB_EXIT_HIGH)
    pattern_breakdown = (bb_test < BB_EXIT_LOW) & (macd_test < 0)
    exit_pattern = pattern_overheat | pattern_breakdown

    # 게이팅 마스크 (AFTER 기준 — bias 보정된 예측)
    em_gated = np.minimum(np.abs(pred_max) * CONF_MULT, 1.0) > ENTRY_THR
    xm_gated = np.minimum(np.abs(pred_min) * CONF_MULT, 1.0) > EXIT_THR
    em_filtered = em_gated & entry_pattern
    xm_filtered = xm_gated & exit_pattern

    n_entry_filt = int(em_filtered.sum())
    n_exit_filt = int(xm_filtered.sum())
    da_entry_filt = float((np.sign(pred_max[em_filtered]) == np.sign(actual_max[em_filtered])).mean() * 100) if n_entry_filt > 0 else None
    da_exit_filt = float((np.sign(pred_min[xm_filtered]) == np.sign(actual_min[xm_filtered])).mean() * 100) if n_exit_filt > 0 else None
    if da_entry_filt is not None and da_exit_filt is not None:
        da_filt_avg = (da_entry_filt + da_exit_filt) / 2
    elif da_entry_filt is not None:
        da_filt_avg = da_entry_filt
    elif da_exit_filt is not None:
        da_filt_avg = da_exit_filt
    else:
        da_filt_avg = None

    n_test = len(X_test)
    entry_keep_rate = round(float(n_entry_gated / n_test * 100), 1) if n_test else 0.0
    exit_keep_rate = round(float(n_exit_gated / n_test * 100), 1) if n_test else 0.0
    entry_keep_rate_nb = round(float(n_entry_nb / n_test * 100), 1) if n_test else 0.0
    exit_keep_rate_nb = round(float(n_exit_nb / n_test * 100), 1) if n_test else 0.0

    delta_bias = (
        round(float(da_gated_avg - da_gated_nb_avg), 2)
        if (da_gated_avg is not None and da_gated_nb_avg is not None) else None
    )

    # MODIFIED [Multi-Filter]: 다중 필터 keep rate / delta 계산
    entry_keep_rate_filt = round(float(n_entry_filt / n_test * 100), 1) if n_test else 0.0
    exit_keep_rate_filt = round(float(n_exit_filt / n_test * 100), 1) if n_test else 0.0
    delta_filter = (
        round(float(da_filt_avg - da_gated_avg), 2)
        if (da_filt_avg is not None and da_gated_avg is not None) else None
    )

    return {
        'code': code,
        'n_test': n_test,
        'systematic_bias': round(systematic_bias, 6),
        # 참고용 — per-step DA (bias 적용 후)
        'da_t1': round(float(da_t1), 2),
        'da_t2': round(float(da_t2), 2),
        'da_t3': round(float(da_t3), 2),
        'mean_comp_DA': round(float(da_mean_comp), 2),
        # max/min 메트릭 (bias 적용 후) — 게이팅 전 단계 참조
        'da_entry_after_bias': round(float(da_entry), 2),
        'da_exit_after_bias': round(float(da_exit), 2),
        'max_min_DA_after_bias': round(float(da_max_min_avg), 2),
        # MODIFIED [Systematic Bias]: BEFORE = bias 없는 게이팅 결과
        'BEFORE_n_entry_gated': n_entry_nb,
        'BEFORE_n_exit_gated': n_exit_nb,
        'BEFORE_entry_keep_rate': entry_keep_rate_nb,
        'BEFORE_exit_keep_rate': exit_keep_rate_nb,
        'BEFORE_da_entry_gated': round(da_entry_nb, 2) if da_entry_nb is not None else None,
        'BEFORE_da_exit_gated': round(da_exit_nb, 2) if da_exit_nb is not None else None,
        'BEFORE_gated_DA': round(da_gated_nb_avg, 2) if da_gated_nb_avg is not None else None,
        # MODIFIED [Systematic Bias]: AFTER = bias 적용된 게이팅 결과
        'AFTER_n_entry_gated': n_entry_gated,
        'AFTER_n_exit_gated': n_exit_gated,
        'AFTER_entry_keep_rate': entry_keep_rate,
        'AFTER_exit_keep_rate': exit_keep_rate,
        'AFTER_da_entry_gated': round(da_entry_gated, 2) if da_entry_gated is not None else None,
        'AFTER_da_exit_gated': round(da_exit_gated, 2) if da_exit_gated is not None else None,
        'AFTER_gated_DA': round(da_gated_avg, 2) if da_gated_avg is not None else None,
        'delta': delta_bias,
        # MODIFIED [Multi-Filter]: 게이팅 + 다중 필터 (이번 단계 신규 메트릭)
        'FILTER_n_entry': n_entry_filt,
        'FILTER_n_exit': n_exit_filt,
        'FILTER_entry_keep_rate': entry_keep_rate_filt,
        'FILTER_exit_keep_rate': exit_keep_rate_filt,
        'FILTER_da_entry': round(da_entry_filt, 2) if da_entry_filt is not None else None,
        'FILTER_da_exit': round(da_exit_filt, 2) if da_exit_filt is not None else None,
        'FILTER_DA': round(da_filt_avg, 2) if da_filt_avg is not None else None,
        'delta_filter': delta_filter,
    }


def _fmt_pct(v):
    return f"{v:6.2f}%" if v is not None else "  N/A "


def main():
    print("=" * 120)
    print(f"[Multi-Filter] gating(mult={CONF_MULT}, e={ENTRY_THR}, x={EXIT_THR}) "
          f"+ RSI/BB/MACD AND 패턴 적용 | BEFORE(gating only) vs AFTER(gating+filter)")  # MODIFIED [Multi-Filter]
    print(f"  진입 패턴: rebound(rsi<{RSI_ENTRY_LOW}, bb<{BB_ENTRY_LOW}) OR breakout(rsi>{RSI_ENTRY_HIGH}, bb>{BB_ENTRY_HIGH}, macd>0)")
    print(f"  청산 패턴: overheat(rsi>{RSI_EXIT_HIGH}, bb>{BB_EXIT_HIGH}) OR breakdown(bb<{BB_EXIT_LOW}, macd<0)")
    print("=" * 120)

    results = []
    for code in CODES:
        print(f"\n[측정] {code} ...", flush=True)
        try:
            r = measure_one(code)
            results.append(r)
            # MODIFIED [Multi-Filter]: BEFORE = 이번 단계의 비교 기준 = 이전 단계 AFTER_gated_DA
            before_str = f"{r['AFTER_gated_DA']:.2f}%" if r['AFTER_gated_DA'] is not None else "N/A"
            after_str = f"{r['FILTER_DA']:.2f}%" if r['FILTER_DA'] is not None else "N/A"
            delta_str = f"{r['delta_filter']:+.2f}p" if r['delta_filter'] is not None else "N/A"
            print(f"  N_test={r['n_test']} | "
                  f"BEFORE(gating): {before_str} | AFTER(+filter): {after_str} | Δ {delta_str}")
            print(f"    BEFORE keep entry={r['AFTER_entry_keep_rate']}% / exit={r['AFTER_exit_keep_rate']}% "
                  f"|| AFTER keep entry={r['FILTER_entry_keep_rate']}% / exit={r['FILTER_exit_keep_rate']}%")
        except Exception as e:
            import traceback
            print(f"  [ERROR] {code}: {e}")
            traceback.print_exc()

    # MODIFIED [Multi-Filter]: 표 출력 — BEFORE(gating only) vs AFTER(gating + multi-filter)
    print("\n" + "=" * 120)
    print("종목별 결과 (Multi-Filter)")
    print("=" * 120)
    print(f"{'Code':<8}{'N_test':>7}  "
          f"{'BEFORE':>9}{'B.EKeep':>9}{'B.XKeep':>9}  "
          f"{'AFTER':>9}{'A.EKeep':>9}{'A.XKeep':>9}  "
          f"{'Δ':>8}")
    print("-" * 120)
    for r in results:
        before = _fmt_pct(r['AFTER_gated_DA'])
        after = _fmt_pct(r['FILTER_DA'])
        delta = f"{r['delta_filter']:+6.2f}p" if r['delta_filter'] is not None else "  N/A "
        print(f"{r['code']:<8}{r['n_test']:>7}  "
              f"{before:>9}{r['AFTER_entry_keep_rate']:>8.1f}%{r['AFTER_exit_keep_rate']:>8.1f}%  "
              f"{after:>9}{r['FILTER_entry_keep_rate']:>8.1f}%{r['FILTER_exit_keep_rate']:>8.1f}%  "
              f"{delta:>8}")

    if results:
        valid_before = [r['AFTER_gated_DA'] for r in results if r['AFTER_gated_DA'] is not None]
        valid_after = [r['FILTER_DA'] for r in results if r['FILTER_DA'] is not None]
        avg_before = np.mean(valid_before) if valid_before else None
        avg_after = np.mean(valid_after) if valid_after else None
        avg_keep_entry_b = np.mean([r['AFTER_entry_keep_rate'] for r in results])
        avg_keep_exit_b = np.mean([r['AFTER_exit_keep_rate'] for r in results])
        avg_keep_entry_a = np.mean([r['FILTER_entry_keep_rate'] for r in results])
        avg_keep_exit_a = np.mean([r['FILTER_exit_keep_rate'] for r in results])

        print("-" * 120)
        before_str = _fmt_pct(avg_before)
        after_str = _fmt_pct(avg_after)
        if avg_before is not None and avg_after is not None:
            delta_str = f"{avg_after - avg_before:+6.2f}p"
        else:
            delta_str = "  N/A "
        print(f"{'평균':<8}{'':>7}  "
              f"{before_str:>9}{avg_keep_entry_b:>8.1f}%{avg_keep_exit_b:>8.1f}%  "
              f"{after_str:>9}{avg_keep_entry_a:>8.1f}%{avg_keep_exit_a:>8.1f}%  "
              f"{delta_str:>8}")
        print("=" * 120)
        if avg_before is not None and avg_after is not None:
            print(f"\n>>> 5종목 평균 방향정확도: BEFORE(gating) {avg_before:.2f}% → AFTER(+filter) {avg_after:.2f}% (Δ {avg_after - avg_before:+.2f}p)")
            print(f">>> 평균 거래 빈도(keep): BEFORE entry {avg_keep_entry_b:.1f}% / exit {avg_keep_exit_b:.1f}%  ||  AFTER entry {avg_keep_entry_a:.1f}% / exit {avg_keep_exit_a:.1f}%")

    # JSON 저장
    out_path = os.path.join(SCRIPT_DIR, 'measure_da_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n결과 JSON: {out_path}")


if __name__ == '__main__':
    main()