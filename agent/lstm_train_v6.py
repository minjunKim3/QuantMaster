"""
LSTM V6 — V5 (Foundation 앙상블) 의 학습 시간 단축 버전.

변경점 (vs V5):
  [B] AutoGluon 에서 Chronos 1개만 학습/예측  (DeepAR/ADIDA 제거)
       → autogluon phase 약 132s → 약 20s 단축
저장 위치:  agent/models_v6/

NOTE 20260602: 초기 C(epoch 40/patience 10) 적용했더니 일부 종목(000660)
방향정확도가 25pp 하락. Confidence Gating 은 정상 동작 중이지만 학습 자체가
부족했던 게 원인. → C 제거, LSTM 학습 설정은 V5 와 동일(100 epoch, patience 20) 유지.

목표:    종목당 학습 시간 약 80~120초 (V5 약 195초 대비 단축)
영향:    수익률 타겟 / 방향 가중 손실 / Systematic Bias / 멀티필터 임계값 그대로 유지.
         lstm_service.py inference 가 fm_deepar / fm_adida 를 0 으로 패딩하므로
         학습 시점에 그 두 컬럼이 없어도 V5 와 동일한 inference 호환성 유지.
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import FinanceDataReader as fdr
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
import pywt
import copy
import time
import os
import pickle
import yfinance as yf
from datetime import datetime

from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

# V5 의 PhaseTimer / set_seed / _seed_worker / DWT / external / huber / directional_huber
# 그리고 멀티필터 임계값을 그대로 가져온다 (재구현 비용 0, 동작 동일성 보장).
from lstm_train_v5 import (
    PhaseTimer,
    set_seed,
    _seed_worker,
    add_dwt_features,
    get_external_data,
    huber_loss,
    directional_huber_loss,
    create_features_v4,
    RSI_ENTRY_LOW, BB_ENTRY_LOW,
    RSI_ENTRY_HIGH, BB_ENTRY_HIGH,
    RSI_EXIT_HIGH, BB_EXIT_HIGH, BB_EXIT_LOW,
    SEED,
)


# ============================================
# 모델 정의 — V5 와 동일 구조 (이름만 분리)
# ============================================
class StockLSTM_V6(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 3)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm1d,)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def generate_foundation_predictions_chronos_only(df, code, close_mean, model_dir, timer=None):
    print("  [Foundation/V6] AutoGluon (Chronos + DeepAR + ADIDA) 학습 시작...")

    close_normalized = df['Close'].values / close_mean

    ts_df = pd.DataFrame({
        'item_id': 'STOCK',
        'timestamp': df.index.tz_localize(None) if df.index.tz else df.index,
        'target': close_normalized
    })
    ts_df = TimeSeriesDataFrame(ts_df)

    safe_code = code.replace('^', '').replace('/', '_')
    ag_path = os.path.join(model_dir, f'{safe_code}_autogluon')

    t_loadfit = time.time()
    loaded_existing = False
    predictor = None
    if os.path.exists(ag_path):
        try:
            predictor = TimeSeriesPredictor.load(ag_path)
            loaded_existing = True
            print("  [Foundation/V6] 기존 AutoGluon 모델 로드 성공")
        except Exception:
            print("  [Foundation/V6] 로드 실패, 새로 학습")
            predictor = None

    if predictor is None:
        predictor = TimeSeriesPredictor(
            prediction_length=1,
            cache_predictions=False,
            log_to_file=False,
            path=ag_path,
            freq='D'
        )
        predictor.fit(
            ts_df,
            enable_ensemble=False,
            hyperparameters={
                'Chronos': {"model_path": "amazon/chronos-bolt-small"},
                # autogluon 1.1+ 은 epochs 가 아니라 max_epochs 만 받음
                'DeepAR': {
                    'max_epochs': 8,
                    'num_layers': 1,
                    'hidden_size': 20,
                    'context_length': 30,
                },
                'ADIDA': {},
            },
            time_limit=120
        )
        predictor.save()
        print("  [Foundation/V6] AutoGluon 학습 완료")
    if timer is not None:
        label = "ag_load" if loaded_existing else "ag_fit"
        timer.add(label, time.time() - t_loadfit, parent="autogluon")

    MODEL_NAMES = {
        'Chronos[amazon__chronos-bolt-small]': 'chronos',
        'DeepAR': 'deepar',
        'ADIDA': 'adida',
    }
    predictions = {name: np.full(len(df), np.nan) for name in MODEL_NAMES.values()}

    context_len = 60
    batch_size = 1024

    for model_name, col_name in MODEL_NAMES.items():
        print(f"  [Foundation/V6] {model_name} rolling 예측")
        t_model = time.time()

        for batch_start in range(context_len, len(df), batch_size):
            batch_end = min(batch_start + batch_size, len(df))
            batch_ts_list = []
            for i in range(batch_start, batch_end):
                window_data = df.iloc[i - context_len:i]
                ts_window = pd.DataFrame({
                    'item_id': f'DATA_{i}',
                    'timestamp': window_data.index.tz_localize(None) if window_data.index.tz else window_data.index,
                    'target': window_data['Close'].values / close_mean
                })
                batch_ts_list.append(ts_window)
            batch_ts = pd.concat(batch_ts_list, ignore_index=True)

            try:
                ts_data = TimeSeriesDataFrame.from_data_frame(
                    batch_ts, id_column='item_id', timestamp_column='timestamp'
                )
                pred = predictor.predict(ts_data, model=model_name)
                for idx in range(batch_start, batch_end):
                    item_id = f'DATA_{idx}'
                    try:
                        predictions[col_name][idx] = pred.loc[item_id]['mean'].iloc[0]
                    except Exception:
                        pass
            except Exception as e:
                print(f"    배치 {batch_start}-{batch_end} 실패: {e}")
                continue

        if timer is not None:
            timer.add(f"rolling_{col_name}", time.time() - t_model, parent="autogluon")
        valid_count = np.isfinite(predictions[col_name]).sum()
        print(f"  [Foundation/V6] {model_name} 완료: {valid_count}개 예측")

    return predictions


# ============================================
# 학습 함수
# ============================================
# 20260602: LSTM epoch 루프 시간 예산 (초). 누적 학습 시간이 이 값 넘으면 즉시 중단.
#   patience-based early stop 과 OR 조건으로 동작.
#   초안 120s = 2분. AutoGluon (16~46s) + LSTM (≤120s) = 종목당 약 2~3분 목표.
LSTM_TIME_BUDGET_S = 90   # 20260603: 120→90 (LGBM 30s 추가로 종합 3분컷 유지)


def train_model(code, start_date='2015-01-01', timer=None):
    set_seed(SEED)
    print(f"\n{'='*60}")
    print(f"LSTM V6 학습: {code} (Chronos only + LSTM 100ep/patience20 + time budget {LSTM_TIME_BUDGET_S}s)")
    print(f"{'='*60}")

    own_timer = timer is None
    if own_timer:
        timer = PhaseTimer()

    # 1. 데이터
    print(f"\n[1] 데이터 다운로드: {code}")
    with timer.phase("data_download"):
        df = fdr.DataReader(code, start_date)
    if df.empty or len(df) < 200:
        print(f"  데이터 부족! ({len(df)}일)")
        return False
    print(f"  데이터: {len(df)}일")

    close_mean = df['Close'].mean()
    volume_mean = df['Volume'].mean()

    # 2. Foundation (Chronos only)
    print(f"\n[2] Foundation Model 예측 (Chronos only)")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, 'models_v6')
    os.makedirs(model_dir, exist_ok=True)

    with timer.phase("autogluon"):
        foundation_preds = generate_foundation_predictions_chronos_only(
            df, code, close_mean, model_dir, timer=timer
        )

    # 3. 외부 지표
    print(f"\n[3] 외부 지표 로드")
    with timer.phase("external"):
        external_data = get_external_data(start_date, df.index[-1].strftime('%Y-%m-%d'), timer=timer)

    # 4. 특성 (V5 와 동일한 create_features_v4 재사용. fm_deepar / fm_adida 미포함 시 자동 누락.)
    print(f"\n[4] 특성 생성")
    with timer.phase("features"):
        features = create_features_v4(df, external_data, foundation_preds)
    print(f"  특성 수: {len(features.columns)}개")
    print(f"  특성 목록: {list(features.columns)}")

    # 5. 정규화
    with timer.phase("lstm_prep"):
        scaler = RobustScaler()
        target_scaler = RobustScaler()

        feature_values = features.values
        scaled = scaler.fit_transform(feature_values)

        WINDOW = 60
        close_raw = df['Close'].values

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
        y_scaled = target_scaler.fit_transform(y)

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y_scaled[:split], y_scaled[split:]
        y_test_raw = y[split:]

        print(f"  학습: {len(X_train)}개 | 테스트: {len(X_test)}개")

    # 6. 학습 [C] EPOCHS 40, patience 10
    print(f"\n[5] 학습 시작")
    lstm_train_ctx = timer.phase("lstm_train")
    lstm_train_ctx.__enter__()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StockLSTM_V6(input_size=X.shape[-1]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    train_dataset = TensorDataset(torch.FloatTensor(X_train).to(device), torch.FloatTensor(y_train).to(device))
    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True,
        worker_init_fn=_seed_worker, generator=g
    )
    val_dataset = TensorDataset(torch.FloatTensor(X_test).to(device), torch.FloatTensor(y_test).to(device))
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False,
        worker_init_fn=_seed_worker
    )

    # budget 만료 시 즉시 중단하지 않고, 평균 DA 측정 → 목표 미달이면 연장.
    # 약한 모델에 시간을 더 주되 최대 (INITIAL + EXTENSION × MAX_EXTENSIONS) 로 cap.
    EPOCHS = 300
    EARLY_STOP_PATIENCE = 20
    INITIAL_BUDGET_S = LSTM_TIME_BUDGET_S
    EXTENSION_BUDGET_S = 30
    MAX_EXTENSIONS = 2
    DA_TARGET = 55.0
    DA_PLATEAU_DELTA = 0.5
    current_budget = INITIAL_BUDGET_S
    extension_count = 0
    last_check_da_avg = None

    # 반복 평가 시 매 호출 GPU 복사 비용 피하려고 한 번만 옮김
    X_train_eval_t = torch.FloatTensor(X_train).to(device)
    X_test_eval_t = torch.FloatTensor(X_test).to(device)

    def _quick_da_avg():
        """현재 model 의 test set 평균 DA (max/min trick, bias 보정 포함). 반환: (avg, entry, exit)."""
        model.eval()
        with torch.no_grad():
            pr_tr = model(X_train_eval_t).cpu().numpy()
            pr_te = model(X_test_eval_t).cpu().numpy()
        pr_tr_raw = target_scaler.inverse_transform(pr_tr)
        pr_te_raw = target_scaler.inverse_transform(pr_te)
        y_tr_raw_full = target_scaler.inverse_transform(y_train)
        bias = float(y_tr_raw_full.mean() - pr_tr_raw.mean())
        pr_te_corr = pr_te_raw + bias
        pmax = pr_te_corr.max(axis=1)
        pmin = pr_te_corr.min(axis=1)
        amax = y_test_raw.max(axis=1)
        amin = y_test_raw.min(axis=1)
        da_e = float((np.sign(pmax) == np.sign(amax)).mean() * 100.0)
        da_x = float((np.sign(pmin) == np.sign(amin)).mean() * 100.0)
        return (da_e + da_x) / 2.0, da_e, da_x

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    start_time = time.time()
    budget_stopped = False
    iterative_stop_reason = None

    def _check_and_decide(epoch_n, trigger):
        nonlocal current_budget, extension_count, iterative_stop_reason, budget_stopped, patience_counter, last_check_da_avg
        if best_state is not None:
            model.load_state_dict(best_state)
        da_avg, da_e, da_x = _quick_da_avg()
        elapsed_now = time.time() - start_time
        print(f"  [Iter] epoch {epoch_n}, {trigger} (누적 {elapsed_now:.0f}s) → "
              f"Entry {da_e:.1f}% / Exit {da_x:.1f}% / 평균 {da_avg:.1f}%")
        if da_avg >= DA_TARGET:
            iterative_stop_reason = f"target_met_via_{trigger}"
            print(f"  [Iter] 평균 DA {da_avg:.1f}% ≥ {DA_TARGET}% → 학습 종료")
            budget_stopped = True
            return True
        # 첫 체크는 비교 대상 없음 → 연장 1회는 무조건 허용
        if last_check_da_avg is not None:
            delta = da_avg - last_check_da_avg
            if delta < DA_PLATEAU_DELTA:
                iterative_stop_reason = f"plateau_via_{trigger}"
                print(f"  [Iter] 직전 대비 DA 변화 {delta:+.2f}pp < {DA_PLATEAU_DELTA}pp → plateau 감지, 학습 종료")
                budget_stopped = True
                return True
        if extension_count >= MAX_EXTENSIONS:
            iterative_stop_reason = f"max_extensions_via_{trigger}"
            print(f"  [Iter] 평균 DA {da_avg:.1f}% < {DA_TARGET}% 이지만 최대 연장 {MAX_EXTENSIONS}회 도달 → 학습 종료")
            budget_stopped = True
            return True
        extension_count += 1
        # patience 케이스는 아직 시간 남았어도 elapsed 기준으로 잡아야 진짜로 그만큼 더 받음
        current_budget = max(current_budget + EXTENSION_BUDGET_S, elapsed_now + EXTENSION_BUDGET_S)
        patience_counter = 0
        last_check_da_avg = da_avg
        print(f"  [Iter] 평균 DA {da_avg:.1f}% < {DA_TARGET}% → +{EXTENSION_BUDGET_S}s 연장 "
              f"#{extension_count}/{MAX_EXTENSIONS} (누적 budget {current_budget:.0f}s, patience 리셋)")
        return False

    for epoch in range(EPOCHS):
        elapsed_so_far = time.time() - start_time
        if elapsed_so_far > current_budget:
            if _check_and_decide(epoch + 1, "budget"):
                break

        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = directional_huber_loss(pred, batch_y, delta=0.01, direction_weight=3.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                pred = model(batch_x)
                loss = directional_huber_loss(pred, batch_y, delta=0.01, direction_weight=3.0)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | {elapsed:.0f}초 (budget {current_budget:.0f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                if _check_and_decide(epoch + 1, "patience"):
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

    if best_state:
        model.load_state_dict(best_state)

    elapsed = time.time() - start_time
    stop_reason = iterative_stop_reason or ("patience" if patience_counter >= EARLY_STOP_PATIENCE else "max_epochs")
    print(f"  학습 완료! {elapsed:.0f}초 ({elapsed/60:.1f}분) | stop_reason={stop_reason} | 연장 {extension_count}/{MAX_EXTENSIONS}")
    lstm_train_ctx.__exit__(None, None, None)

    # 7. 평가 (Systematic Bias 보정 포함)
    print(f"\n[6] 평가")
    lstm_eval_ctx = timer.phase("lstm_eval")
    lstm_eval_ctx.__enter__()
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()
    pred_raw = target_scaler.inverse_transform(pred_scaled)

    with torch.no_grad():
        pred_train_scaled = model(torch.FloatTensor(X_train).to(device)).cpu().numpy()
    pred_train_raw = target_scaler.inverse_transform(pred_train_scaled)
    y_train_raw_full = target_scaler.inverse_transform(y_train)
    systematic_bias = float(y_train_raw_full.mean() - pred_train_raw.mean())
    print(f"  [Systematic Bias] mean(actual)={y_train_raw_full.mean():.6f} | mean(pred)={pred_train_raw.mean():.6f} | bias={systematic_bias:+.6f}")

    pred_raw_uncorrected = pred_raw.copy()
    pred_raw = pred_raw + systematic_bias

    for step in range(3):
        actual = y_test_raw[:, step]
        predicted = pred_raw[:, step]
        actual_price = (1 + actual) * close_mean
        predicted_price = (1 + predicted) * close_mean
        mape = np.mean(np.abs((actual_price - predicted_price) / actual_price)) * 100
        rmse = np.sqrt(mean_squared_error(actual_price, predicted_price))
        print(f"  t+{step+1}: MAPE {mape:.2f}% | RMSE {rmse:.2f}")
        direction_correct = np.sign(y_test_raw[:, step]) == np.sign(pred_raw[:, step])
        direction_acc = direction_correct.mean() * 100
        print(f"  t+{step+1}: Direction Accuracy {direction_acc:.2f}%")

    # Max/Min 트릭 DA
    pred_max = pred_raw.max(axis=1)
    actual_max = y_test_raw.max(axis=1)
    da_entry = (np.sign(pred_max) == np.sign(actual_max)).mean() * 100
    pred_min = pred_raw.min(axis=1)
    actual_min = y_test_raw.min(axis=1)
    da_exit = (np.sign(pred_min) == np.sign(actual_min)).mean() * 100
    da_max_min_avg = (da_entry + da_exit) / 2
    print(f"  [max-min ] Entry DA: {da_entry:.2f}% | Exit DA: {da_exit:.2f}% | 평균: {da_max_min_avg:.2f}%")

    # Confidence Gating (raw 수익률 단위; V5 와 동일 thresholds)
    GATE_MULT = 200.0
    GATE_ENTRY_THR = 0.21
    GATE_EXIT_THR = 0.32
    conf_entry = np.minimum(np.abs(pred_max) * GATE_MULT, 1.0)
    conf_exit = np.minimum(np.abs(pred_min) * GATE_MULT, 1.0)
    entry_mask = conf_entry > GATE_ENTRY_THR
    exit_mask = conf_exit > GATE_EXIT_THR
    n_test = len(pred_raw)
    n_entry_gated = int(entry_mask.sum())
    n_exit_gated = int(exit_mask.sum())
    da_entry_gated = (np.sign(pred_max[entry_mask]) == np.sign(actual_max[entry_mask])).mean() * 100 if n_entry_gated > 0 else None
    da_exit_gated = (np.sign(pred_min[exit_mask]) == np.sign(actual_min[exit_mask])).mean() * 100 if n_exit_gated > 0 else None
    da_gated_avg = None
    if da_entry_gated is not None and da_exit_gated is not None:
        da_gated_avg = (da_entry_gated + da_exit_gated) / 2
    entry_da_str = f"{da_entry_gated:.2f}%" if da_entry_gated is not None else "N/A"
    exit_da_str = f"{da_exit_gated:.2f}%" if da_exit_gated is not None else "N/A"
    avg_str = f"{da_gated_avg:.2f}%" if da_gated_avg is not None else "N/A"
    print(f"  [GATED   ] Entry DA: {entry_da_str} (keep {n_entry_gated}/{n_test}) | "
          f"Exit DA: {exit_da_str} (keep {n_exit_gated}/{n_test}) | 평균: {avg_str}")

    # Multi-Filter DA
    test_now_idx = split + WINDOW
    rsi_test = features['rsi'].values[test_now_idx:test_now_idx + len(X_test)]
    bb_test = features['bb_position'].values[test_now_idx:test_now_idx + len(X_test)]
    macd_test = features['macd_hist'].values[test_now_idx:test_now_idx + len(X_test)]
    pat_rebound = (rsi_test < RSI_ENTRY_LOW) & (bb_test < BB_ENTRY_LOW)
    pat_breakout = (rsi_test > RSI_ENTRY_HIGH) & (bb_test > BB_ENTRY_HIGH) & (macd_test > 0)
    pat_overheat = (rsi_test > RSI_EXIT_HIGH) & (bb_test > BB_EXIT_HIGH)
    pat_breakdown = (bb_test < BB_EXIT_LOW) & (macd_test < 0)
    em_filt = entry_mask & (pat_rebound | pat_breakout)
    xm_filt = exit_mask & (pat_overheat | pat_breakdown)
    n_e_filt = int(em_filt.sum())
    n_x_filt = int(xm_filt.sum())
    da_e_filt = (np.sign(pred_max[em_filt]) == np.sign(actual_max[em_filt])).mean() * 100 if n_e_filt > 0 else None
    da_x_filt = (np.sign(pred_min[xm_filt]) == np.sign(actual_min[xm_filt])).mean() * 100 if n_x_filt > 0 else None
    da_filt_avg = None
    if da_e_filt is not None and da_x_filt is not None:
        da_filt_avg = (da_e_filt + da_x_filt) / 2
    e_filt_str = f"{da_e_filt:.2f}%" if da_e_filt is not None else "N/A"
    x_filt_str = f"{da_x_filt:.2f}%" if da_x_filt is not None else "N/A"
    avg_filt_str = f"{da_filt_avg:.2f}%" if da_filt_avg is not None else "N/A"
    print(f"  [FILTER  ] Entry DA: {e_filt_str} (keep {n_e_filt}/{n_test}) | "
          f"Exit DA: {x_filt_str} (keep {n_x_filt}/{n_test}) | 평균: {avg_filt_str}")

    # Joint calibration: Entry/Exit (e_r, x_r) grid search.
    # 평균 통과율 (e_r + x_r) / 2 가 TARGET ± BAND 안인 후보만 본 다음, 가중 DA 최대 선택.
    # 범위 안 후보가 없으면 (entry=100%, exit=30%) fallback.
    CANDIDATE_RATIOS_ENTRY = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    CANDIDATE_RATIOS_EXIT  = [0.20, 0.30, 0.40, 0.50, 0.60]
    TARGET_AVG_PASS_RATE   = 0.625
    TARGET_PASS_BAND       = 0.075
    MIN_SAMPLES = 30
    TIE_BAND_DA = 0.005

    def _joint_calibrate(pmax, amax, pmin, amin):
        n = len(pmax)
        abs_max = np.abs(pmax)
        abs_min = np.abs(pmin)
        e_baseline = float((np.sign(pmax) == np.sign(amax)).mean())
        x_baseline = float((np.sign(pmin) == np.sign(amin)).mean())
        cands = []
        for e_r in CANDIDATE_RATIOS_ENTRY:
            if e_r >= 1.0:
                e_thr = 0.0
                e_mask = np.ones(n, dtype=bool)
            else:
                e_thr = float(np.quantile(abs_max, 1.0 - e_r))
                e_mask = abs_max > e_thr
            e_n = int(e_mask.sum())
            if e_n < MIN_SAMPLES:
                continue
            e_da = float((np.sign(pmax[e_mask]) == np.sign(amax[e_mask])).mean())
            e_ar = e_n / n
            for x_r in CANDIDATE_RATIOS_EXIT:
                if x_r >= 1.0:
                    x_thr = 0.0
                    x_mask = np.ones(n, dtype=bool)
                else:
                    x_thr = float(np.quantile(abs_min, 1.0 - x_r))
                    x_mask = abs_min > x_thr
                x_n = int(x_mask.sum())
                if x_n < MIN_SAMPLES:
                    continue
                x_da = float((np.sign(pmin[x_mask]) == np.sign(amin[x_mask])).mean())
                x_ar = x_n / n
                avg_pass = (e_ar + x_ar) / 2.0
                if not (TARGET_AVG_PASS_RATE - TARGET_PASS_BAND <= avg_pass <= TARGET_AVG_PASS_RATE + TARGET_PASS_BAND):
                    continue
                w_da = (e_ar * e_da + x_ar * x_da) / (e_ar + x_ar)
                cands.append((w_da, abs(avg_pass - TARGET_AVG_PASS_RATE),
                              e_thr, e_da, e_ar, x_thr, x_da, x_ar, avg_pass))
        if not cands:
            x_thr_fb = float(np.quantile(abs_min, 0.70))
            x_mask_fb = abs_min > x_thr_fb
            x_da_fb = float((np.sign(pmin[x_mask_fb]) == np.sign(amin[x_mask_fb])).mean()) if x_mask_fb.sum() else x_baseline
            x_ar_fb = float(x_mask_fb.sum()) / n
            avg_fb = (1.0 + x_ar_fb) / 2.0
            print(f"  [CAL Joint] 후보 없음 → fallback entry=100% exit=30% (avg_pass={avg_fb*100:.1f}%)")
            return (0.0, e_baseline, 1.0, "fallback", e_baseline,
                    x_thr_fb, x_da_fb, x_ar_fb, "fallback", x_baseline)
        # 1차 정렬: 가중 DA 큰 것 / 2차: 평균 통과율이 TARGET 에 가까운 것
        best_w = max(c[0] for c in cands)
        near = [c for c in cands if c[0] >= best_w - TIE_BAND_DA]
        near.sort(key=lambda c: c[1])
        b = near[0]
        print(f"  [CAL Joint] entry_r={b[4]*100:.1f}% entry_DA={b[3]*100:.2f}% (baseline {e_baseline*100:.2f}%) | "
              f"exit_r={b[7]*100:.1f}% exit_DA={b[6]*100:.2f}% (baseline {x_baseline*100:.2f}%) | "
              f"avg_pass={b[8]*100:.1f}% | weighted_DA={b[0]*100:.2f}%")
        e_label = f"joint_e{int(round(b[4]*100))}"
        x_label = f"joint_x{int(round(b[7]*100))}"
        return (b[2], b[3], b[4], e_label, e_baseline,
                b[5], b[6], b[7], x_label, x_baseline)

    (entry_thr_raw, gate_entry_train_da, gate_entry_train_ratio, entry_tgt, entry_baseline,
     exit_thr_raw, gate_exit_train_da, gate_exit_train_ratio, exit_tgt, exit_baseline) = \
        _joint_calibrate(pred_max, actual_max, pred_min, actual_min)

    gate_entry_thr_raw = entry_thr_raw
    gate_exit_thr_raw = exit_thr_raw

    lstm_eval_ctx.__exit__(None, None, None)

    # 8. 저장
    print(f"\n[7] 모델 저장")
    lstm_save_ctx = timer.phase("lstm_save")
    lstm_save_ctx.__enter__()
    safe_code = code.replace('^', '').replace('/', '_')

    model_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    trained_at_iso = datetime.now().isoformat(timespec='seconds')
    prefix = f'{safe_code}_{model_id}'
    torch.save(model.state_dict(), os.path.join(model_dir, f'{prefix}_lstm_v6.pth'))

    # 통과 표본 수로 가중평균한 단일 DA — UI 의 "전체 방향정확도" 표시용
    _denom = float(gate_entry_train_ratio + gate_exit_train_ratio)
    if _denom > 0:
        gated_weighted_da_pct = float(
            (gate_entry_train_ratio * gate_entry_train_da
             + gate_exit_train_ratio * gate_exit_train_da) / _denom * 100.0
        )
    else:
        gated_weighted_da_pct = float(da_max_min_avg)

    # 자신감 = 새 예측의 percentile rank — 게이팅 컷(=1-train_ratio)과 같은 단위로 비교 가능
    abs_max_sorted = np.sort(np.abs(pred_max)).astype(float).tolist()
    abs_min_sorted = np.sort(np.abs(pred_min)).astype(float).tolist()

    meta = {
        'input_size': X.shape[-1],
        'model_id': model_id,
        'trained_at': trained_at_iso,
        'gated_weighted_da_pct': gated_weighted_da_pct,
        'confidence_abs_max_dist': abs_max_sorted,
        'confidence_abs_min_dist': abs_min_sorted,
        'feature_columns': list(features.columns),
        'close_mean': close_mean,
        'volume_mean': volume_mean,
        'window': WINDOW,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'version': 'v6',
        'target_type': 'return',
        'direction_weight': 3.0,
        'systematic_bias': systematic_bias,
        'filter_rsi_entry_low': RSI_ENTRY_LOW,
        'filter_bb_entry_low': BB_ENTRY_LOW,
        'filter_rsi_entry_high': RSI_ENTRY_HIGH,
        'filter_bb_entry_high': BB_ENTRY_HIGH,
        'filter_rsi_exit_high': RSI_EXIT_HIGH,
        'filter_bb_exit_high': BB_EXIT_HIGH,
        'filter_bb_exit_low': BB_EXIT_LOW,
        'foundation_models': ['chronos', 'deepar', 'adida'],
        # 평가용 부가 정보 (서비스 응답에는 미사용; 비교 분석에 활용)
        'eval_da_entry_max_min': float(da_entry),
        'eval_da_exit_max_min': float(da_exit),
        'eval_da_max_min_avg': float(da_max_min_avg),
        'eval_da_gated_avg': float(da_gated_avg) if da_gated_avg is not None else None,
        'eval_da_filter_avg': float(da_filt_avg) if da_filt_avg is not None else None,
        # thr=0 + ratio=1.0 이면 그 종목엔 게이팅이 도움 안 됐다는 의미
        'gate_calibrated': True,
        'gate_entry_thr_raw': float(gate_entry_thr_raw),
        'gate_exit_thr_raw': float(gate_exit_thr_raw),
        'gate_entry_train_da': float(gate_entry_train_da),
        'gate_exit_train_da': float(gate_exit_train_da),
        'gate_entry_train_ratio': float(gate_entry_train_ratio),
        'gate_exit_train_ratio': float(gate_exit_train_ratio),
        'gate_entry_baseline_da': float(entry_baseline),
        'gate_exit_baseline_da': float(exit_baseline),
        'gate_entry_target_label': str(entry_tgt),  # "target_60" 또는 "no_gating_unhelpful"
        'gate_exit_target_label': str(exit_tgt),
        'gate_calibration_method': 'dynamic_v1',
        # 20260603 ITERATIVE: 반복 학습 메타
        'train_initial_budget_s': INITIAL_BUDGET_S,
        'train_extension_budget_s': EXTENSION_BUDGET_S,
        'train_max_extensions': MAX_EXTENSIONS,
        'train_da_target': DA_TARGET,
        'train_extensions_used': extension_count,
        'train_final_budget_s': current_budget,
        'train_stop_reason': stop_reason,
    }

    with open(os.path.join(model_dir, f'{prefix}_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(model_dir, f'{prefix}_target_scaler.pkl'), 'wb') as f:
        pickle.dump(target_scaler, f)
    with open(os.path.join(model_dir, f'{prefix}_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    # Java/UI 에서 읽기 쉬운 JSON 요약본 — feature_columns/scaler 등 큰 객체는 제외
    _meta_json = {
        'modelId': model_id,
        'trainedAt': trained_at_iso,
        'version': 'v6',
        'code': code,
        'gatedWeightedDaPct': gated_weighted_da_pct,
        'evalDaEntryMaxMin': float(da_entry),
        'evalDaExitMaxMin': float(da_exit),
        'evalDaMaxMinAvg': float(da_max_min_avg),
        'gateEntryThrRaw': float(gate_entry_thr_raw),
        'gateExitThrRaw': float(gate_exit_thr_raw),
        'gateEntryTrainDa': float(gate_entry_train_da),
        'gateExitTrainDa': float(gate_exit_train_da),
        'gateEntryTrainRatio': float(gate_entry_train_ratio),
        'gateExitTrainRatio': float(gate_exit_train_ratio),
        'trainStopReason': stop_reason,
        'trainExtensionsUsed': extension_count,
        'trainFinalBudgetS': float(current_budget),
        'foundationModels': ['chronos', 'deepar', 'adida'],
    }
    with open(os.path.join(model_dir, f'{prefix}_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(_meta_json, f, ensure_ascii=False, indent=2, default=str)

    print(f"  저장 완료: {model_dir}/{prefix}_* (modelId={model_id})")
    lstm_save_ctx.__exit__(None, None, None)

    if own_timer:
        timer.report(f"{code}")
    return True


# ============================================
# 메인
# ============================================
if __name__ == '__main__':
    targets = {
        'KS11':   '코스피 지수',
        'KQ11':   '코스닥 지수',
        '005930': '삼성전자',
        '000660': 'SK하이닉스',
        '035720': '카카오',
    }

    print("=" * 60)
    print("LSTM V6 — Chronos only + LSTM 40 epoch / patience 10 (학습 시간 단축)")
    print(f"대상: {len(targets)}개 종목")
    print("=" * 60)

    total_start = time.time()
    success = 0
    per_ticker = []

    for code, name in targets.items():
        print(f"\n>>> {name} ({code})")
        timer = PhaseTimer()
        if train_model(code, timer=timer):
            success += 1
        total = timer.report(f"{name} ({code})")
        per_ticker.append((f"{name}({code})", timer, total))

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"전체 완료! {success}/{len(targets)}개 성공")
    print(f"총 소요시간: {total_elapsed/60:.1f}분")
    print(f"{'='*60}")

    if per_ticker:
        phases = ["data_download", "autogluon", "external", "features",
                  "lstm_prep", "lstm_train", "lstm_eval", "lstm_save"]
        print(f"\n{'='*60}")
        print(f"  종목 간 PHASE 비교 (단위: 초)")
        print(f"{'='*60}")
        header = f"  {'phase':<16}" + "".join(f"{n[:14]:>15}" for n, _, _ in per_ticker)
        print(header)
        print("  " + "-" * (16 + 15 * len(per_ticker)))
        for ph in phases:
            row = f"  {ph:<16}"
            for _, tm, _ in per_ticker:
                sec = sum(s for n, p, s in tm.records if n == ph and p is None)
                row += f"{sec:>14.1f}s"
            print(row)
        print("  " + "-" * (16 + 15 * len(per_ticker)))
        row = f"  {'TOTAL':<16}"
        for _, _, tot in per_ticker:
            row += f"{tot:>14.1f}s"
        print(row)
        print(f"{'='*60}")
