"""
V5 LSTM Hyperopt — Optuna TPE 기반 후처리 임계값 최적화 (5종목 일괄)

목적:
  학습된 V5 모델 가중치는 그대로 두고, 매도 신호(exit) 후처리 임계값만
  Optuna TPE로 종목별 탐색해 DA_exit를 최대화.

목적 함수:
  maximize: avg_DA_exit - λ × Σ max(0, min_trades - actual_exits)²
  - avg_DA_exit  : 종목별 매도 방향정확도(%)의 평균
  - min_trades   : 종목당 연 10회 (val 길이에 비례 조정)
  - λ            : 0.001 (시작값, 추후 조정 가능)

탐색 공간 (5개, ETH 단위 — 내부에서 우리 BB 단위로 변환: our_bb = (eth_bb + 1) / 2):
  - conf_exit     : Float(0.005, 0.70) # 하한 대폭 확장
  - conf_mult     : Float(100, 3000, log) # 종목별 변동성 자동 보정
  - bb_exit_high  : Float(0.5, 1.5)    # 현재 1.16 (= 우리 1.08)
  - bb_exit_low   : Float(-1.0, -0.3)  # 현재 -0.77 (= 우리 0.115)
  - rsi_exit_high : Int(55, 80)        # 현재 65

데이터 분할 (80/10/10):
  - Train 0-80%   : 이미 학습 완료, 건드림 X
  - Val   80-90%  : TPE 탐색
  - Test  90-100% : Hold-out, Best params로 1회 평가

시드:
  - 학습 SEED=42 (고정)
  - Optuna sampler seed = SEED + ticker_idx (종목별 분리)
"""
import os
import sys
import json
import pickle
import random
import time
import csv
import gc

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import torch
import FinanceDataReader as fdr
import optuna
from optuna.samplers import TPESampler

# ============================================
# 시드 고정 (V5 학습/measure_da.py와 동일)
# ============================================
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

# ============================================
# 설정
# ============================================
TICKERS = ['KS11', 'KQ11', '005930', '000660', '035720']   # V5 weights 보유 5종목
WINDOW = 60
START_DATE = '2015-01-01'
MODEL_DIR = os.path.join(SCRIPT_DIR, 'models_v5')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'hyperopt_results')
N_TRIALS = 100                          # 1-2 단계 결과: trial 수가 시간에 거의 영향 없음 → 안전 마진 확보
WARN_AT_MINUTES = 80                    # 80분 경과 시 경고

# 고정 하이퍼파라미터
LAMBDA_PENALTY = 0.001           # 페널티 강도 (사용자 권고 시작값)
MIN_TRADES_PER_YEAR = 10         # 종목당 연 최소 매도 신호 수 (도메인 지식)
TRADING_DAYS_PER_YEAR = 252

# Entry 임계값 고정 — 매수는 잘 돌고 있어서 건드리지 않음
RSI_ENTRY_LOW = 38.0
BB_ENTRY_LOW = 0.285
RSI_ENTRY_HIGH = 31.0
BB_ENTRY_HIGH = 0.51


# ============================================
# 데이터 준비 (종목별 1회만 실행, trial마다 재사용)
# ============================================
def prepare_data(code: str) -> dict:
    """학습된 V5 모델로 val/test 예측 + 보조 지표를 미리 계산해서 캐싱."""
    safe_code = code.replace('^', '').replace('/', '_')

    print(f"[준비] {code} 데이터/모델 로드...", flush=True)

    # 1. 원본 데이터
    df = fdr.DataReader(code, START_DATE)
    if df.empty or len(df) < 200:
        raise RuntimeError(f"{code}: 데이터 부족 ({len(df)}일)")
    close_mean = df['Close'].mean()

    # 2. Foundation Model 예측 (캐시 hit 기대 — 학습된 AutoGluon 디렉토리 로드)
    foundation_preds = generate_foundation_predictions(df, code, close_mean, MODEL_DIR)

    # 3. 외부 지표 + 특성
    external_data = get_external_data(START_DATE, df.index[-1].strftime('%Y-%m-%d'))
    features = create_features_v4(df, external_data, foundation_preds)

    # 4. 저장된 scaler/meta 로드
    with open(os.path.join(MODEL_DIR, f'{safe_code}_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, f'{safe_code}_target_scaler.pkl'), 'rb') as f:
        target_scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, f'{safe_code}_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)

    expected_cols = meta['feature_columns']
    for col in expected_cols:
        if col not in features.columns:
            features[col] = 0
    features = features[expected_cols]

    scaled = scaler.transform(features.values)
    close_raw = df['Close'].values

    # 5. 윈도우 + 수익률 타겟 (학습 시와 동일 로직)
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

    # 6. 80/10/10 분할 (사령관 권고)
    n_total = len(X)
    train_end = int(n_total * 0.8)
    val_end = int(n_total * 0.9)

    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

    # 7. 모델 로드
    input_size = X.shape[-1]
    model = StockLSTM_V5(input_size=input_size)
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, f'{safe_code}_lstm_v5.pth'),
        map_location='cpu', weights_only=True
    ))
    model.eval()

    # 8. 추론 (train/val/test 한 번씩만)
    with torch.no_grad():
        pred_train_scaled = model(torch.FloatTensor(X_train)).numpy()
        pred_val_scaled = model(torch.FloatTensor(X_val)).numpy()
        pred_test_scaled = model(torch.FloatTensor(X_test)).numpy()

    pred_train_raw = target_scaler.inverse_transform(pred_train_scaled)
    pred_val_raw = target_scaler.inverse_transform(pred_val_scaled)
    pred_test_raw = target_scaler.inverse_transform(pred_test_scaled)

    # 9. Systematic bias (train으로 계산, val/test에 동일 적용)
    systematic_bias = float(y_train.mean() - pred_train_raw.mean())
    pred_val_raw = pred_val_raw + systematic_bias
    pred_test_raw = pred_test_raw + systematic_bias

    # 10. 보조 지표 슬라이스 (val/test 범위)
    val_now_start = train_end + WINDOW
    test_now_start = val_end + WINDOW
    rsi_vals = features['rsi'].values
    bb_vals = features['bb_position'].values
    macd_vals = features['macd_hist'].values

    rsi_val = rsi_vals[val_now_start:val_now_start + len(X_val)]
    bb_val = bb_vals[val_now_start:val_now_start + len(X_val)]
    macd_val = macd_vals[val_now_start:val_now_start + len(X_val)]
    rsi_test = rsi_vals[test_now_start:test_now_start + len(X_test)]
    bb_test = bb_vals[test_now_start:test_now_start + len(X_test)]
    macd_test = macd_vals[test_now_start:test_now_start + len(X_test)]

    # 11. min_trades 비례 조정 (val/test가 1년 미만이면 줄어듦)
    min_trades_val = max(1, round(MIN_TRADES_PER_YEAR * len(X_val) / TRADING_DAYS_PER_YEAR))
    min_trades_test = max(1, round(MIN_TRADES_PER_YEAR * len(X_test) / TRADING_DAYS_PER_YEAR))

    print(f"  데이터: 총 {n_total}샘플 → Train {len(X_train)} / Val {len(X_val)} / Test {len(X_test)}", flush=True)
    print(f"  systematic_bias: {systematic_bias:+.6f}", flush=True)
    print(f"  min_trades 요구치: Val {min_trades_val} / Test {min_trades_test} (연 {MIN_TRADES_PER_YEAR}회 비례)", flush=True)

    return {
        'code': code,
        'pred_val': pred_val_raw, 'y_val': y_val,
        'pred_test': pred_test_raw, 'y_test': y_test,
        'rsi_val': rsi_val, 'bb_val': bb_val, 'macd_val': macd_val,
        'rsi_test': rsi_test, 'bb_test': bb_test, 'macd_test': macd_test,
        'min_trades_val': min_trades_val,
        'min_trades_test': min_trades_test,
        'systematic_bias': systematic_bias,
    }


# ============================================
# Exit 신호 평가 — val/test 공통
# ============================================
def evaluate_exit(pred, y_actual, rsi, bb, macd,
                  conf_exit_thr, conf_mult, bb_exit_high_eth, bb_exit_low_eth, rsi_exit_high):
    """매도 신호의 (DA_exit, n_exits) 반환. ETH 단위 → 우리 BB 단위 변환."""
    bb_exit_high_our = (bb_exit_high_eth + 1) / 2
    bb_exit_low_our = (bb_exit_low_eth + 1) / 2

    pred_min = pred.min(axis=1)
    actual_min = y_actual.min(axis=1)

    # Confidence gating (conf_mult도 trial별로 탐색)
    conf_exit = np.minimum(np.abs(pred_min) * conf_mult, 1.0)
    xm_gated = conf_exit > conf_exit_thr

    # Multi-filter pattern (overheat OR breakdown)
    pattern_overheat = (rsi > rsi_exit_high) & (bb > bb_exit_high_our)
    pattern_breakdown = (bb < bb_exit_low_our) & (macd < 0)
    exit_pattern = pattern_overheat | pattern_breakdown

    xm_filtered = xm_gated & exit_pattern
    n_exits = int(xm_filtered.sum())

    if n_exits == 0:
        return 0.0, 0
    da_exit = float((np.sign(pred_min[xm_filtered]) == np.sign(actual_min[xm_filtered])).mean() * 100)
    return da_exit, n_exits


# ============================================
# Optuna objective
# ============================================
def make_objective(prepared_list):
    def objective(trial: optuna.Trial) -> float:
        conf_exit_thr = trial.suggest_float('conf_exit', 0.005, 0.70)
        conf_mult = trial.suggest_float('conf_mult', 100.0, 3000.0, log=True)
        bb_exit_high_eth = trial.suggest_float('bb_exit_high', 0.5, 1.5)
        bb_exit_low_eth = trial.suggest_float('bb_exit_low', -1.0, -0.3)
        rsi_exit_high = trial.suggest_int('rsi_exit_high', 55, 80)

        da_list, n_exits_list, min_trades_list = [], [], []
        for p in prepared_list:
            da, n_exits = evaluate_exit(
                p['pred_val'], p['y_val'],
                p['rsi_val'], p['bb_val'], p['macd_val'],
                conf_exit_thr, conf_mult, bb_exit_high_eth, bb_exit_low_eth, rsi_exit_high,
            )
            da_list.append(da)
            n_exits_list.append(n_exits)
            min_trades_list.append(p['min_trades_val'])

        avg_da = float(np.mean(da_list))
        penalty = sum(
            max(0, mn - n) ** 2
            for n, mn in zip(n_exits_list, min_trades_list)
        )
        score = avg_da - LAMBDA_PENALTY * penalty

        trial.set_user_attr('avg_da', avg_da)
        trial.set_user_attr('n_exits_per_ticker', n_exits_list)
        trial.set_user_attr('penalty', penalty)
        return score
    return objective


# ============================================
# 1종목 실행 + 즉시 저장
# ============================================
def run_ticker(code: str, ticker_idx: int, total: int) -> dict:
    """1종목 hyperopt 실행 → 종목별 best_params.json / trials_log.csv 즉시 저장 → 결과 dict 반환."""
    t_start = time.time()
    print(f"\n{'#' * 80}")
    print(f"# [{ticker_idx + 1}/{total}] {code} 시작")
    print(f"{'#' * 80}\n", flush=True)

    # 1) 데이터/모델 준비
    p = prepare_data(code)
    t_prep_end = time.time()

    # 2) Optuna study (종목별 sampler seed 분리)
    sampler_seed = SEED + ticker_idx
    sampler = TPESampler(seed=sampler_seed)
    study = optuna.create_study(
        direction='maximize', sampler=sampler,
        study_name=f'v5_hyperopt_{code}',
    )
    print(f"\n[Optuna] {code} TPE sampler(seed={sampler_seed}) {N_TRIALS} trials...", flush=True)
    study.optimize(make_objective([p]), n_trials=N_TRIALS, show_progress_bar=False)
    t_opt_end = time.time()

    best = study.best_trial
    bp = best.params

    # 3) Test hold-out 평가
    da_t, n_t = evaluate_exit(
        p['pred_test'], p['y_test'],
        p['rsi_test'], p['bb_test'], p['macd_test'],
        bp['conf_exit'], bp['conf_mult'], bp['bb_exit_high'], bp['bb_exit_low'], bp['rsi_exit_high'],
    )

    # 4) 결과 패키징
    safe_code = code.replace('^', '').replace('/', '_')
    result = {
        'code': code,
        'sampler_seed': sampler_seed,
        'n_trials': N_TRIALS,
        'best_trial': best.number,
        'best_score': round(float(best.value), 4),
        'best_params': bp,
        'bb_our_units': {
            'bb_exit_high_our': round((bp['bb_exit_high'] + 1) / 2, 4),
            'bb_exit_low_our': round((bp['bb_exit_low'] + 1) / 2, 4),
        },
        'val': {
            'avg_da': round(best.user_attrs.get('avg_da', 0.0), 2),
            'n_exits': best.user_attrs.get('n_exits_per_ticker', [0])[0],
            'min_trades_required': p['min_trades_val'],
            'penalty': best.user_attrs.get('penalty', 0),
        },
        'test': {
            'da_exit': round(da_t, 2),
            'n_exits': n_t,
            'min_trades_required': p['min_trades_test'],
        },
        'systematic_bias': round(p['systematic_bias'], 6),
        'timing_sec': {
            'prep': round(t_prep_end - t_start, 1),
            'optuna': round(t_opt_end - t_prep_end, 1),
            'total': round(t_opt_end - t_start, 1),
        },
    }

    # 5) 즉시 저장 (중단 대비)
    best_path = os.path.join(OUTPUT_DIR, f'{safe_code}_best_params.json')
    with open(best_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(OUTPUT_DIR, f'{safe_code}_trials_log.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['trial', 'score', 'avg_da', 'n_exits', 'penalty',
                    'conf_exit', 'conf_mult', 'bb_exit_high', 'bb_exit_low', 'rsi_exit_high'])
        for t in study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            nex = t.user_attrs.get('n_exits_per_ticker', [0])
            params = t.params
            w.writerow([
                t.number,
                round(float(t.value), 4) if t.value is not None else None,
                round(t.user_attrs.get('avg_da', 0.0), 2),
                nex[0] if nex else 0,
                t.user_attrs.get('penalty', 0),
                round(params.get('conf_exit', 0.0), 6),
                round(params.get('conf_mult', 0.0), 2),
                round(params.get('bb_exit_high', 0.0), 4),
                round(params.get('bb_exit_low', 0.0), 4),
                params.get('rsi_exit_high', 0),
            ])

    print(f"\n  → Val DA={result['val']['avg_da']:.2f}% (신호 {result['val']['n_exits']}개, min {result['val']['min_trades_required']})")
    print(f"  → Test DA={result['test']['da_exit']:.2f}% (신호 {result['test']['n_exits']}개, hold-out)")
    print(f"  → Val→Test 갭: {result['val']['avg_da'] - result['test']['da_exit']:+.2f}p")
    print(f"  → 소요: 준비 {result['timing_sec']['prep']:.1f}s + Optuna {result['timing_sec']['optuna']:.1f}s = {result['timing_sec']['total']:.1f}s")
    print(f"  → 저장: {best_path}")
    print(f"  → 저장: {csv_path}")

    # 6) 메모리 해제 (다음 종목 OOM 방지)
    del p, study
    gc.collect()

    return result


# ============================================
# Summary 마크다운 생성
# ============================================
def write_summary(all_results: list, total_elapsed: float, out_path: str):
    lines = []
    lines.append("# V5 LSTM Hyperopt — 5종목 일괄 결과\n")
    lines.append(f"- 생성: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 종목 수: {len(all_results)} / 요청 {len(TICKERS)}")
    lines.append(f"- 종목당 trial: {N_TRIALS}")
    lines.append(f"- 학습 SEED: {SEED} (고정) | Sampler seed: SEED + ticker_idx (종목별 분리)")
    lines.append(f"- λ(penalty): {LAMBDA_PENALTY} | min_trades/year: {MIN_TRADES_PER_YEAR}")
    lines.append(f"- 총 소요: {total_elapsed:.1f}초 ({total_elapsed/60:.2f}분)\n")

    if not all_results:
        lines.append("⚠️ 성공한 종목 없음 — 로그 확인 필요\n")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        return

    # 전체 요약 표
    lines.append("## 전체 요약\n")
    lines.append("| 종목 | Val DA | Val 신호 | Test DA | Test 신호 | Δ갭 | conf_exit | conf_mult | bb_high | bb_low | rsi_high | 시간(초) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in all_results:
        bp = r['best_params']
        gap = r['val']['avg_da'] - r['test']['da_exit']
        lines.append(
            f"| {r['code']} "
            f"| {r['val']['avg_da']:.2f}% | {r['val']['n_exits']} "
            f"| {r['test']['da_exit']:.2f}% | {r['test']['n_exits']} "
            f"| {gap:+.2f}p "
            f"| {bp['conf_exit']:.4f} | {bp['conf_mult']:.1f} "
            f"| {bp['bb_exit_high']:.4f} | {bp['bb_exit_low']:.4f} | {bp['rsi_exit_high']} "
            f"| {r['timing_sec']['total']:.1f} |"
        )
    lines.append("")

    # 평균
    val_das = [r['val']['avg_da'] for r in all_results]
    test_das = [r['test']['da_exit'] for r in all_results]
    avg_val = sum(val_das) / len(val_das)
    avg_test = sum(test_das) / len(test_das)
    lines.append("## 평균\n")
    lines.append(f"- 5종목 **Val** DA 평균: **{avg_val:.2f}%**")
    lines.append(f"- 5종목 **Test** (hold-out) DA 평균: **{avg_test:.2f}%**")
    lines.append(f"- Val→Test 갭: **{avg_val - avg_test:+.2f}p** (>10p면 과적합 신호)\n")

    # 종목별 상세
    lines.append("## 종목별 상세\n")
    for r in all_results:
        lines.append(f"### {r['code']}\n")
        lines.append(f"- Sampler seed: {r['sampler_seed']}")
        lines.append(f"- Best trial: #{r['best_trial']} (score={r['best_score']:.4f})")
        lines.append(f"- Systematic bias: {r['systematic_bias']:+.6f}")
        lines.append(f"- BB 임계값 (우리 단위): high={r['bb_our_units']['bb_exit_high_our']}, low={r['bb_our_units']['bb_exit_low_our']}")
        lines.append(f"- Val: DA {r['val']['avg_da']:.2f}% / 신호 {r['val']['n_exits']}개 (min {r['val']['min_trades_required']}) / penalty {r['val']['penalty']}")
        lines.append(f"- Test (hold-out): DA {r['test']['da_exit']:.2f}% / 신호 {r['test']['n_exits']}개 (min {r['test']['min_trades_required']})")
        lines.append(f"- 시간: 준비 {r['timing_sec']['prep']:.1f}s + Optuna {r['timing_sec']['optuna']:.1f}s = {r['timing_sec']['total']:.1f}s\n")

    # 산출물
    lines.append("## 산출물\n")
    for r in all_results:
        safe_code = r['code'].replace('^', '').replace('/', '_')
        lines.append(f"- `{safe_code}_best_params.json` / `{safe_code}_trials_log.csv`")
    lines.append("")

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# ============================================
# 메인
# ============================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print(f"V5 LSTM Hyperopt — {len(TICKERS)}종목 일괄")
    print(f"종목     : {TICKERS}")
    print(f"Trial    : {N_TRIALS}/종목")
    print(f"학습 SEED: {SEED} (고정) | Sampler seed = SEED + ticker_idx (종목별 분리)")
    print(f"고정     : λ={LAMBDA_PENALTY}, min_trades/year={MIN_TRADES_PER_YEAR}")
    print(f"탐색(5)  : conf_exit, conf_mult, bb_exit_high, bb_exit_low, rsi_exit_high")
    print(f"출력     : {OUTPUT_DIR}")
    print("=" * 80, flush=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)  # 종목 5개 × 100 trial 로그 폭주 방지

    t_global = time.time()
    all_results = []

    for idx, code in enumerate(TICKERS):
        try:
            r = run_ticker(code, idx, len(TICKERS))
            all_results.append(r)
        except Exception as e:
            import traceback
            print(f"\n  ❌ [ERROR] {code}: {e}", flush=True)
            traceback.print_exc()
            continue

        # 진행률 + 80분 가드
        elapsed = time.time() - t_global
        avg_per = elapsed / (idx + 1)
        remaining_est = avg_per * (len(TICKERS) - idx - 1)
        print(f"\n  [진행률 {idx + 1}/{len(TICKERS)}] 누적 {elapsed:.0f}s ({elapsed/60:.1f}분) | 잔여 추정 {remaining_est:.0f}s ({remaining_est/60:.1f}분)", flush=True)
        if elapsed > WARN_AT_MINUTES * 60:
            print(f"  ⚠️ {WARN_AT_MINUTES}분 경과! 남은 {len(TICKERS) - idx - 1}종목 진행 중 — 필요시 수동 중단", flush=True)

    total_elapsed = time.time() - t_global

    # Summary 생성
    summary_path = os.path.join(OUTPUT_DIR, 'summary.md')
    write_summary(all_results, total_elapsed, summary_path)

    # 콘솔 최종 표
    print("\n" + "=" * 80)
    print(f"[완료] {len(all_results)}/{len(TICKERS)}종목 | 총 {total_elapsed:.1f}초 ({total_elapsed/60:.2f}분)")
    print(f"Summary: {summary_path}")
    print("=" * 80)

    if all_results:
        print(f"\n[5종목 요약]")
        print(f"{'종목':<8} {'Val DA':>9} {'Test DA':>9} {'갭':>7} {'시간(s)':>9}")
        print("-" * 50)
        vavg = tavg = 0.0
        for r in all_results:
            vda, tda = r['val']['avg_da'], r['test']['da_exit']
            print(f"{r['code']:<8} {vda:>8.2f}% {tda:>8.2f}% {vda - tda:>+6.2f}p {r['timing_sec']['total']:>9.1f}")
            vavg += vda
            tavg += tda
        n = len(all_results)
        print("-" * 50)
        print(f"{'평균':<8} {vavg/n:>8.2f}% {tavg/n:>8.2f}% {(vavg - tavg)/n:>+6.2f}p")


if __name__ == '__main__':
    main()