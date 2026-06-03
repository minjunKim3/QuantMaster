import sys
import json
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import FinanceDataReader as fdr
from sklearn.preprocessing import RobustScaler
import pywt
import pickle
import os
import yfinance as yf

# 20260602: 새 종목 자동학습이 10분 가까이 걸리는 원인 추적용 단순 phase 타이머.
# - 학습 내부는 lstm_train_v5.py 의 PhaseTimer 가 따로 출력.
# - 여기서는 service 레벨에서 import / data_dl / auto_train / predict_setup / predict_loop / postprocess 만 잰다.
# - 출력은 stderr (Java BacktestRunService 가 `{`로 시작 안 하는 라인을 [LSTM] 로 로깅).
_SVC_PHASES = []          # [(name, sec)]

class _SvcPhase:
    def __init__(self, name: str):
        self.name = name
        self.t0 = 0.0
    def __enter__(self):
        self.t0 = time.time()
        return self
    def __exit__(self, *a):
        _SVC_PHASES.append((self.name, time.time() - self.t0))

def _svc_phase(name: str):
    return _SvcPhase(name)

def _svc_phase_report(header: str):
    if not _SVC_PHASES:
        return
    # __overall__ 같은 메타 phase 는 총합 % 계산에서 제외 (다른 phase 들과 겹치므로)
    summable = [(n, s) for n, s in _SVC_PHASES if not n.startswith("__")]
    overall = dict(_SVC_PHASES).get("__overall__")
    total = sum(s for _, s in summable)
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  TIMING (service) - {header}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    for n, s in _SVC_PHASES:
        if n.startswith("__"):
            continue
        pct = (s / total * 100) if total > 0 else 0
        print(f"  {n:<22} {s:>7.1f}s ({pct:5.1f}%)", file=sys.stderr)
    print(f"  {'-'*55}", file=sys.stderr)
    print(f"  SUM of phases          {total:>7.1f}s ({total/60:.2f}min)", file=sys.stderr)
    if overall is not None:
        print(f"  OVERALL wall (main)    {overall:>7.1f}s ({overall/60:.2f}min)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

# ============================================
# 재현성: 시드 고정 (예측 결과 재현 보장)
# ============================================
SEED = 42

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 모듈 로드 시 시드 고정
set_seed(SEED)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

# ============================================
# V3 / V5 / V6 공용 모델 정의 — 구조 동일 (lstm_train_v3/v5/v6.py 와 일치)
#   autogluon import 회피 위해 service 측에 재정의해서 load_state_dict 만 호출.
#   각 버전은 학습 설정(loss, target, foundation)만 다르고 forward 구조는 같음.
# 20260602: 이전엔 동일한 클래스 3개를 따로 선언해서 약 80줄 중복 → 별칭으로 정리.
# ============================================
class StockLSTM_V3(nn.Module):
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

# V5/V6 도 동일 구조 → 별칭으로 사용 (학습 설정만 다름).
StockLSTM_V5 = StockLSTM_V3
StockLSTM_V6 = StockLSTM_V3

# ============================================
# 기존 V2 모델 (폴백용)
# ============================================
class StockLSTM_V2(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        return self.fc2(out).squeeze()

# ============================================
# DWT 웨이블릿
# ============================================
def add_dwt_features(close_prices, window_size=128):
    n = len(close_prices)
    dwt_trend = np.zeros(n)
    dwt_high = np.zeros(n)
    dwt_low = np.zeros(n)
    dwt_energy = np.zeros(n)
    for i in range(window_size, n):
        window = close_prices[i - window_size:i]
        try:
            coeffs = pywt.wavedec(window, 'db4', level=2)
            dwt_trend[i] = coeffs[0][-1]
            dwt_high[i] = coeffs[1][-1]
            dwt_low[i] = coeffs[2][-1]
            dwt_energy[i] = np.sum(coeffs[1]**2) + np.sum(coeffs[2]**2)
        except Exception:
            pass
    return dwt_trend, dwt_high, dwt_low, dwt_energy

# ============================================
# 특성 생성 (V3)
# ============================================
def create_features_v3(df, external_data=None):
    close = df['Close'].values
    volume = df['Volume'].values
    
    features = pd.DataFrame(index=df.index)
    features['close'] = close
    features['ma5'] = pd.Series(close).rolling(5).mean().values
    features['ma20'] = pd.Series(close).rolling(20).mean().values
    
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss_val = (-delta.where(delta < 0, 0)).rolling(14).mean()
    features['rsi'] = (100 - (100 / (1 + gain / (loss_val + 1e-10)))).values
    features['change'] = pd.Series(close).pct_change().values
    features['volume_norm'] = volume / (np.max(volume) + 1e-10)
    
    ma20_s = pd.Series(close).rolling(20).mean()
    std20 = pd.Series(close).rolling(20).std()
    bb_upper = ma20_s + 2 * std20
    bb_lower = ma20_s - 2 * std20
    features['bb_position'] = ((close - bb_lower) / (bb_upper - bb_lower + 1e-8)).values
    
    exp12 = pd.Series(close).ewm(span=12).mean()
    exp26 = pd.Series(close).ewm(span=26).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9).mean()
    features['macd_hist'] = (macd - signal).values
    
    high = df['High'].values if 'High' in df.columns else close
    low = df['Low'].values if 'Low' in df.columns else close
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    features['atr'] = pd.Series(tr).rolling(14).mean().values
    
    dwt_trend, dwt_high, dwt_low, dwt_energy = add_dwt_features(close, min(128, len(close) // 4))
    features['dwt_trend'] = dwt_trend
    features['dwt_high'] = dwt_high
    features['dwt_low'] = dwt_low
    features['dwt_energy'] = dwt_energy
    
    if external_data:
        for name, series in external_data.items():
            aligned = series.reindex(df.index, method='ffill')
            features[f'{name}_norm'] = (aligned / aligned.mean()).values
    
    features = features.ffill().bfill().fillna(0)
    return features

# ============================================
# 외부 지표
# ============================================
def get_external_data(start_date, end_date):
    externals = {}
    symbols = {'vix': '^VIX', 'gold': 'GC=F', 'dxy': 'DX-Y.NYB'}
    for name, ticker in symbols.items():
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                externals[name] = data['Close']
        except Exception:
            pass
    return externals

# ============================================
# 메인 실행
# ============================================
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({"error": "파라미터 필요"}, cls=NpEncoder))
        sys.exit(1)
    
    arg = sys.argv[1]
    if arg.endswith('.json'):
        with open(arg, 'r', encoding='utf-8') as f:
            params = json.load(f)
    else:
        params = json.loads(arg)
    
    code = params.get('code', 'KS11')
    days = params.get('days', 100)

    # MODIFIED [Model Version Routing]: UI/백엔드가 명시한 모델 버전 우선 적용
    # - 미지정 / "Auto" / "" → 기존 자동 폴백 (V5 > V4 > V3 > V2)
    # - "V2"~"V5" 명시 + 해당 weights 존재 → 그 모델 강제 사용
    # - 명시 모델 없음 / 잘못된 값 → 자동 폴백 + modelFallback=true 경고
    requested_version_raw = str(params.get('modelVersion') or '').strip().upper()
    if requested_version_raw in ('', 'AUTO', 'NONE', 'NULL'):
        requested_version = None  # 자동 폴백
    else:
        requested_version = requested_version_raw

    code = code.replace('.KS', '').replace('.KQ', '')

    # modelId 명시 → 그 파일 / 미명시 → 최신 timestamp → legacy
    # forceTrain=true → modelId 무시 + 무조건 새 V6 학습
    model_id_req = str(params.get('modelId') or '').strip()
    force_train = bool(params.get('forceTrain', False))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    safe_code = code.replace('^', '').replace('/', '_')

    # MODIFIED V6: V6 모델 확인 (B+C — Chronos only + LSTM 40epoch, 최우선 폴백)
    v6_dir = os.path.join(script_dir, 'models_v6')

    def _resolve_v6_paths(_model_id):
        import glob
        from datetime import datetime as _dt
        if _model_id:
            _p = f'{safe_code}_{_model_id}'
            return (os.path.join(v6_dir, f'{_p}_lstm_v6.pth'),
                    os.path.join(v6_dir, f'{_p}_meta.pkl'),
                    os.path.join(v6_dir, f'{_p}_scaler.pkl'),
                    os.path.join(v6_dir, f'{_p}_target_scaler.pkl'))
        matches = glob.glob(os.path.join(v6_dir, f'{safe_code}_*_lstm_v6.pth'))
        cands = []
        for _m in matches:
            _bn = os.path.basename(_m)
            _core = _bn[:-len('_lstm_v6.pth')]
            if not _core.startswith(safe_code + '_'):
                continue
            _ts = _core[len(safe_code) + 1:]
            try:
                _dt.strptime(_ts, '%Y%m%d_%H%M%S')
                cands.append((_ts, _m))
            except ValueError:
                pass
        if cands:
            cands.sort(reverse=True)
            _ts, _latest = cands[0]
            _p = f'{safe_code}_{_ts}'
            return (_latest,
                    os.path.join(v6_dir, f'{_p}_meta.pkl'),
                    os.path.join(v6_dir, f'{_p}_scaler.pkl'),
                    os.path.join(v6_dir, f'{_p}_target_scaler.pkl'))
        # legacy: timestamp 없는 단일 파일
        return (os.path.join(v6_dir, f'{safe_code}_lstm_v6.pth'),
                os.path.join(v6_dir, f'{safe_code}_meta.pkl'),
                os.path.join(v6_dir, f'{safe_code}_scaler.pkl'),
                os.path.join(v6_dir, f'{safe_code}_target_scaler.pkl'))

    v6_model_path, v6_meta_path, v6_scaler_path, v6_target_scaler_path = _resolve_v6_paths(model_id_req)

    # MODIFIED V5: V5 모델 확인 (수익률 타겟 + 방향 가중 손실)
    v5_dir = os.path.join(script_dir, 'models_v5')
    v5_model_path = os.path.join(v5_dir, f'{safe_code}_lstm_v5.pth')
    v5_meta_path = os.path.join(v5_dir, f'{safe_code}_meta.pkl')
    v5_scaler_path = os.path.join(v5_dir, f'{safe_code}_scaler.pkl')
    v5_target_scaler_path = os.path.join(v5_dir, f'{safe_code}_target_scaler.pkl')

    # V4 모델 확인 (Foundation Model 앙상블)
    v4_dir = os.path.join(script_dir, 'models_v4')
    v4_model_path = os.path.join(v4_dir, f'{safe_code}_lstm_v4.pth')
    v4_meta_path = os.path.join(v4_dir, f'{safe_code}_meta.pkl')
    v4_scaler_path = os.path.join(v4_dir, f'{safe_code}_scaler.pkl')
    v4_target_scaler_path = os.path.join(v4_dir, f'{safe_code}_target_scaler.pkl')
    
    # V3 모델 확인
    v3_dir = os.path.join(script_dir, 'models_v3')
    v3_model_path = os.path.join(v3_dir, f'{safe_code}_lstm_v3.pth')
    v3_meta_path = os.path.join(v3_dir, f'{safe_code}_meta.pkl')
    v3_scaler_path = os.path.join(v3_dir, f'{safe_code}_scaler.pkl')
    v3_target_scaler_path = os.path.join(v3_dir, f'{safe_code}_target_scaler.pkl')
    
    # V2 모델 확인 (폴백)
    v2_dir = os.path.join(script_dir, 'models')
    v2_model_path = os.path.join(v2_dir, f'{safe_code}_lstm.pth')
    
    # force_train=True 면 V6 학습부터 강제 → 다른 모델 fallback 사용 안 함
    use_v6 = (not force_train) and os.path.exists(v6_model_path) and os.path.exists(v6_meta_path)
    use_v5 = (not force_train) and os.path.exists(v5_model_path) and os.path.exists(v5_meta_path)
    use_v4 = (not force_train) and os.path.exists(v4_model_path) and os.path.exists(v4_meta_path)
    use_v3 = (not force_train) and os.path.exists(v3_model_path) and os.path.exists(v3_meta_path)
    use_v2 = (not force_train) and os.path.exists(v2_model_path)

    # 학습된 모델이 하나도 없거나 forceTrain=true 면 즉석 V6 학습
    auto_trained = False
    _svc_overall_t0 = time.time()
    if force_train or (not use_v6 and not use_v5 and not use_v4 and not use_v3 and not use_v2):
        _reason = "강제 학습 요청" if force_train else "학습된 모델 없음"
        print(f"[LSTM-Auto] {code} {_reason} → V6 학습 시작", file=sys.stderr)
        try:
            with _svc_phase("auto_train_import"):
                from lstm_train_v6 import train_model as _train_v6
            with _svc_phase("auto_train"):
                ok = _train_v6(code)
            if not ok:
                print(json.dumps({"error": f"{code} V6 학습 실패 (데이터 부족 또는 잘못된 종목)"}, cls=NpEncoder))
                sys.exit(1)
            # 학습 직후 최신 timestamp 재탐색 → 새 모델로 예측 진행
            v6_model_path, v6_meta_path, v6_scaler_path, v6_target_scaler_path = _resolve_v6_paths('')
            use_v6 = os.path.exists(v6_model_path) and os.path.exists(v6_meta_path)
            if not use_v6:
                print(json.dumps({"error": f"{code} V6 학습 완료 후 모델 파일을 찾지 못함"}, cls=NpEncoder))
                sys.exit(1)
            auto_trained = True
            print(f"[LSTM-Auto] {code} V6 학습 완료 → 예측 진행", file=sys.stderr)
        except Exception as e:
            print(json.dumps({"error": f"{code} V6 자동 학습 중 에러: {str(e)}"}, cls=NpEncoder))
            sys.exit(1)

    # MODIFIED [Model Version Routing]: 사용자가 명시한 버전이 있으면 그것만 활성화, 실패 시 폴백
    # 자동 폴백 메커니즘은 명시 미지정/명시 모델 부재 시 그대로 안전망 역할
    model_fallback = False
    available_map = {'V6': use_v6, 'V5': use_v5, 'V4': use_v4, 'V3': use_v3, 'V2': use_v2}
    if requested_version is not None:
        if requested_version in available_map and available_map[requested_version]:
            # 명시한 모델이 존재 → 그 모델만 사용
            use_v6 = (requested_version == 'V6')
            use_v5 = (requested_version == 'V5')
            use_v4 = (requested_version == 'V4')
            use_v3 = (requested_version == 'V3')
            use_v2 = (requested_version == 'V2')
        else:
            # 명시한 모델이 없거나 잘못된 값 → 폴백 + 경고
            model_fallback = True
            print(f"[LSTM] WARN: 요청 모델 '{requested_version_raw}' 사용 불가 → 자동 폴백 발동",
                  file=sys.stderr)

    # MODIFIED V6: 우선순위 V6 > V5 > V4 > V3 > V2
    model_version = ("V6" if use_v6 else
                     ("V5" if use_v5 else
                      ("V4" if use_v4 else ("V3" if use_v3 else "V2"))))
    print(f"[LSTM] 예측 시작: {code} ({model_version}, requested={requested_version_raw or 'auto'}, fallback={model_fallback})",
          file=sys.stderr)
    
    try:
        # 데이터 다운로드
        with _svc_phase("data_download_predict"):
            df = fdr.DataReader(code, '2015-01-01')

        if use_v6:
            # MODIFIED V6: V6 모델 분기 (B+C — Chronos only + LSTM 40epoch, 수익률 타겟 → 가격 복원)
            #   - 추론 흐름은 V5 와 동일 (24 base 특성 + fm_* 0 패딩).
            print(f"[LSTM] V6 모델 로드 중...", file=sys.stderr)

            with _svc_phase("v6_load_meta_scaler"):
                with open(v6_meta_path, 'rb') as f:
                    meta = pickle.load(f)
                with open(v6_scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                with open(v6_target_scaler_path, 'rb') as f:
                    target_scaler = pickle.load(f)

            close_mean = meta['close_mean']
            WINDOW = meta['window']

            with _svc_phase("v6_external_data"):
                external_data = get_external_data('2015-01-01', df.index[-1].strftime('%Y-%m-%d'))
            with _svc_phase("v6_features"):
                features = create_features_v3(df, external_data)

                expected_cols = meta['feature_columns']
                for col in expected_cols:
                    if col not in features.columns:
                        features[col] = 0  # fm_chronos 등 미존재 시 0 패딩
                features = features[expected_cols]

                scaled = scaler.transform(features.values)

            with _svc_phase("v6_model_load"):
                model = StockLSTM_V6(input_size=meta['input_size'],
                                      hidden_size=meta.get('hidden_size', 128),
                                      num_layers=meta.get('num_layers', 2),
                                      dropout=meta.get('dropout', 0.2))
                model.load_state_dict(torch.load(v6_model_path, map_location='cpu', weights_only=True))
                model.eval()

            predictions_t1 = []
            predictions_t2 = []
            predictions_t3 = []
            actuals = []
            dates = []

            start_idx = max(WINDOW, len(scaled) - days)

            _svc_predict_t0 = time.time()
            for i in range(start_idx, len(scaled) - 3):
                window_data = scaled[i - WINDOW:i]
                X = torch.FloatTensor(window_data).unsqueeze(0)

                with torch.no_grad():
                    pred_scaled = model(X).numpy()

                pred_raw = target_scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]

                bias_v6 = float(meta.get('systematic_bias', 0.0))
                pred_raw = pred_raw + bias_v6

                current_price = float(df['Close'].iloc[i])
                actual_price = float(df['Close'].iloc[i + 1]) if i + 1 < len(df) else current_price

                if meta.get('target_type') == 'return':
                    pred_price_t1 = current_price * (1 + pred_raw[0])
                    pred_price_t2 = current_price * (1 + pred_raw[1])
                    pred_price_t3 = current_price * (1 + pred_raw[2])
                else:
                    pred_price_t1 = pred_raw[0] * close_mean
                    pred_price_t2 = pred_raw[1] * close_mean
                    pred_price_t3 = pred_raw[2] * close_mean

                date = df.index[i].strftime('%Y-%m-%d')
                predictions_t1.append(round(float(pred_price_t1), 2))
                predictions_t2.append(round(float(pred_price_t2), 2))
                predictions_t3.append(round(float(pred_price_t3), 2))
                actuals.append(round(float(actual_price), 2))
                dates.append(date)
            _SVC_PHASES.append(("v6_predict_loop", time.time() - _svc_predict_t0))

        elif use_v5:
            # MODIFIED V5: V5 모델 분기 (수익률 타겟 → 가격 복원)
            print(f"[LSTM] V5 모델 로드 중...", file=sys.stderr)

            with _svc_phase("v5_load_meta_scaler"):
                with open(v5_meta_path, 'rb') as f:
                    meta = pickle.load(f)
                with open(v5_scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                with open(v5_target_scaler_path, 'rb') as f:
                    target_scaler = pickle.load(f)

            close_mean = meta['close_mean']
            WINDOW = meta['window']

            # V5는 V3와 동일한 24개 특성 + Foundation Model 예측을 사용 (학습 시 fm_* 컬럼 포함됨)
            with _svc_phase("v5_external_data"):
                external_data = get_external_data('2015-01-01', df.index[-1].strftime('%Y-%m-%d'))
            with _svc_phase("v5_features"):
                features = create_features_v3(df, external_data)

                expected_cols = meta['feature_columns']
                for col in expected_cols:
                    if col not in features.columns:
                        features[col] = 0  # MODIFIED V5: fm_deepar/fm_chronos/fm_adida 미존재 시 0 패딩
                features = features[expected_cols]

                scaled = scaler.transform(features.values)

            with _svc_phase("v5_model_load"):
                model = StockLSTM_V5(input_size=meta['input_size'],
                                      hidden_size=meta.get('hidden_size', 128),
                                      num_layers=meta.get('num_layers', 2),
                                      dropout=meta.get('dropout', 0.2))
                model.load_state_dict(torch.load(v5_model_path, map_location='cpu', weights_only=True))
                model.eval()

            predictions_t1 = []
            predictions_t2 = []
            predictions_t3 = []
            actuals = []
            dates = []

            start_idx = max(WINDOW, len(scaled) - days)

            _svc_predict_t0 = time.time()
            for i in range(start_idx, len(scaled) - 3):
                window_data = scaled[i - WINDOW:i]
                X = torch.FloatTensor(window_data).unsqueeze(0)

                with torch.no_grad():
                    pred_scaled = model(X).numpy()

                pred_raw = target_scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]

                # MODIFIED [Systematic Bias]: meta에 저장된 bias를 모든 step에 더함 (raw 수익률 단위)
                # 키 없으면 0 → 보정 없이 그대로 사용 (구버전 모델 호환)
                bias_v5 = float(meta.get('systematic_bias', 0.0))
                pred_raw = pred_raw + bias_v5

                current_price = float(df['Close'].iloc[i])
                actual_price = float(df['Close'].iloc[i + 1]) if i + 1 < len(df) else current_price

                # MODIFIED V5: 수익률(return) → 가격 복원
                if meta.get('target_type') == 'return':
                    pred_price_t1 = current_price * (1 + pred_raw[0])
                    pred_price_t2 = current_price * (1 + pred_raw[1])
                    pred_price_t3 = current_price * (1 + pred_raw[2])
                else:
                    # 안전 폴백 (V5 메타가 손상된 경우)
                    pred_price_t1 = pred_raw[0] * close_mean
                    pred_price_t2 = pred_raw[1] * close_mean
                    pred_price_t3 = pred_raw[2] * close_mean

                date = df.index[i].strftime('%Y-%m-%d')

                predictions_t1.append(round(float(pred_price_t1), 2))
                predictions_t2.append(round(float(pred_price_t2), 2))
                predictions_t3.append(round(float(pred_price_t3), 2))
                actuals.append(round(float(actual_price), 2))
                dates.append(date)
            _SVC_PHASES.append(("v5_predict_loop", time.time() - _svc_predict_t0))

        elif use_v4:
            # ===== V4 모델 (Foundation Model 앙상블) =====
            print(f"[LSTM] V4 Foundation Model 로드 중...", file=sys.stderr)

            with open(v4_meta_path, 'rb') as f:
                meta = pickle.load(f)
            with open(v4_scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            with open(v4_target_scaler_path, 'rb') as f:
                target_scaler = pickle.load(f)

            close_mean = meta['close_mean']
            WINDOW = meta['window']

            # 외부 지표
            external_data = get_external_data('2015-01-01', df.index[-1].strftime('%Y-%m-%d'))

            # Foundation Model 예측 생성
            print(f"[LSTM] Foundation Model 예측 생성 중 (5~10분 소요)...", file=sys.stderr)
            from lstm_train_v4 import generate_foundation_predictions, create_features_v4
            foundation_preds = generate_foundation_predictions(df, code, close_mean, v4_dir)

            # 특성 생성
            features = create_features_v4(df, external_data, foundation_preds)

            expected_cols = meta['feature_columns']
            for col in expected_cols:
                if col not in features.columns:
                    features[col] = 0
            features = features[expected_cols]

            scaled = scaler.transform(features.values)
            close_vals = features['close'].values

            # 모델 로드
            from lstm_train_v4 import StockLSTM_V4
            model = StockLSTM_V4(input_size=meta['input_size'],
                                  hidden_size=meta.get('hidden_size', 128),
                                  num_layers=meta.get('num_layers', 2),
                                  dropout=meta.get('dropout', 0.2))
            model.load_state_dict(torch.load(v4_model_path, map_location='cpu', weights_only=True))
            model.eval()

            # 예측
            predictions_t1 = []
            predictions_t2 = []
            predictions_t3 = []
            actuals = []
            dates = []

            start_idx = max(WINDOW, len(scaled) - days)

            for i in range(start_idx, len(scaled) - 3):
                window_data = scaled[i - WINDOW:i]
                X = torch.FloatTensor(window_data).unsqueeze(0)

                with torch.no_grad():
                    pred_scaled = model(X).numpy()

                pred_raw = target_scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]

                current_price = float(close_vals[i])
                actual_price = float(close_vals[i + 1]) if i + 1 < len(close_vals) else current_price

                if meta.get('target_type') == 'returns':
                    pred_price_t1 = current_price * (1 + pred_raw[0])
                    pred_price_t2 = current_price * (1 + pred_raw[1])
                    pred_price_t3 = current_price * (1 + pred_raw[2])
                else:
                    pred_price_t1 = pred_raw[0] * close_mean
                    pred_price_t2 = pred_raw[1] * close_mean
                    pred_price_t3 = pred_raw[2] * close_mean

                date = df.index[i].strftime('%Y-%m-%d')

                predictions_t1.append(round(float(pred_price_t1), 2))
                predictions_t2.append(round(float(pred_price_t2), 2))
                predictions_t3.append(round(float(pred_price_t3), 2))
                actuals.append(round(float(actual_price), 2))
                dates.append(date)
    
        elif use_v3:
            # ===== V3 모델 =====
            with open(v3_meta_path, 'rb') as f:
                meta = pickle.load(f)
            with open(v3_scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            with open(v3_target_scaler_path, 'rb') as f:
                target_scaler = pickle.load(f)
            
            close_mean = meta['close_mean']
            WINDOW = meta['window']
            
            # 외부 지표
            external_data = get_external_data('2015-01-01', df.index[-1].strftime('%Y-%m-%d'))
            
            # 특성 생성
            features = create_features_v3(df, external_data)
            
            # 학습 때와 동일한 특성만 사용
            expected_cols = meta['feature_columns']
            for col in expected_cols:
                if col not in features.columns:
                    features[col] = 0
            features = features[expected_cols]
            
            scaled = scaler.transform(features.values)
            
            # 모델 로드
            model = StockLSTM_V3(input_size=meta['input_size'],
                                  hidden_size=meta.get('hidden_size', 128),
                                  num_layers=meta.get('num_layers', 2),
                                  dropout=meta.get('dropout', 0.2))
            model.load_state_dict(torch.load(v3_model_path, map_location='cpu', weights_only=True))
            model.eval()
            
            # 예측
            close_normalized = features['close'].values / close_mean
            predictions_t1 = []
            predictions_t2 = []
            predictions_t3 = []
            actuals = []
            dates = []
            
            start_idx = max(WINDOW, len(scaled) - days)
            
            for i in range(start_idx, len(scaled) - 3):
                window_data = scaled[i - WINDOW:i]
                X = torch.FloatTensor(window_data).unsqueeze(0)
                
                with torch.no_grad():
                    pred_scaled = model(X).numpy()
                
                pred_raw = target_scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]
                
                current_price = float(df['Close'].iloc[i])
                actual_price = float(df['Close'].iloc[i + 1]) if i + 1 < len(df) else current_price
                
                # 수익률 예측 → 가격으로 변환
                if meta.get('target_type') == 'returns':
                    pred_price_t1 = current_price * (1 + pred_raw[0])
                    pred_price_t2 = current_price * (1 + pred_raw[1])
                    pred_price_t3 = current_price * (1 + pred_raw[2])
                else:
                    pred_price_t1 = pred_raw[0] * close_mean
                    pred_price_t2 = pred_raw[1] * close_mean
                    pred_price_t3 = pred_raw[2] * close_mean
                
                date = df.index[i].strftime('%Y-%m-%d')
                
                predictions_t1.append(round(float(pred_price_t1), 2))
                predictions_t2.append(round(float(pred_price_t2), 2))
                predictions_t3.append(round(float(pred_price_t3), 2))
                actuals.append(round(float(actual_price), 2))
                dates.append(date)
            
        else:
            # ===== V2 모델 (폴백) =====
            close = df['Close']
            features_df = pd.DataFrame(index=df.index)
            features_df['Close'] = close
            features_df['MA5'] = close.rolling(5).mean()
            features_df['MA20'] = close.rolling(20).mean()
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss_val = (-delta.where(delta < 0, 0)).rolling(14).mean()
            features_df['RSI'] = 100 - (100 / (1 + gain / (loss_val + 1e-10)))
            features_df['Change'] = close.pct_change()
            features_df['Volume'] = df['Volume'] / (df['Volume'].max() + 1e-10)
            features_df.ffill(inplace=True)
            features_df.fillna(0, inplace=True)
            
            from sklearn.preprocessing import MinMaxScaler
            scaler_dict = {}
            scaled_df = pd.DataFrame(index=features_df.index)
            for col in features_df.columns:
                s = MinMaxScaler()
                vals = features_df[[col]]
                if vals.max().values[0] == vals.min().values[0]:
                    scaled_df[col] = 0.5
                else:
                    scaled_df[col] = s.fit_transform(vals).flatten()
                scaler_dict[col] = s
            
            close_scaler = scaler_dict['Close']
            scaled = scaled_df.values
            WINDOW = 60
            
            input_size = len(features_df.columns)
            model = StockLSTM_V2(input_size=input_size)
            model.load_state_dict(torch.load(v2_model_path, map_location='cpu', weights_only=True))
            model.eval()
            
            predictions_t1 = []
            actuals = []
            dates = []
            start_idx = max(WINDOW, len(scaled) - days)
            
            for i in range(start_idx, len(scaled)):
                window_data = scaled[i - WINDOW:i]
                X = torch.FloatTensor(window_data).unsqueeze(0)
                with torch.no_grad():
                    pred = model(X).item()
                pred_price = close_scaler.inverse_transform([[pred]])[0][0]
                actual_price = features_df['Close'].iloc[i]
                date = features_df.index[i].strftime('%Y-%m-%d')
                predictions_t1.append(round(float(pred_price), 2))
                actuals.append(round(float(actual_price), 2))
                dates.append(date)

        _svc_postprocess_t0 = time.time()
        # 통계
        pred_arr = np.array(predictions_t1)
        actual_arr = np.array(actuals)
        mape = round(float(np.mean(np.abs((actual_arr - pred_arr) / actual_arr)) * 100), 2)

        pred_diff = np.diff(pred_arr)
        actual_diff = np.diff(actual_arr)
        valid = (pred_diff != 0) & (actual_diff != 0)
        direction = round(float(np.sum(np.sign(pred_diff[valid]) == np.sign(actual_diff[valid])) / valid.sum() * 100), 1) if valid.sum() > 0 else 0

        # MODIFIED 20260601: V5 방향정확도를 lstm_verify.py와 동일한 Confidence Gating 필터 기준으로 통일.
        #   - prev_actual 기준 t+1/t+2/t+3 수익률 → max/min 트릭
        #   - conf = min(|return|*20, 1.0), entry_thr=0.21, exit_thr=0.32
        #   - 통과 표본만 entry/exit DA 계산, 평균 = direction
        if (use_v5 or use_v6) and predictions_t2 and predictions_t3 and len(actual_arr) >= 4:
            # MODIFIED 20260602: MULT 20 → 200 (lstm_train_v5 / lstm_verify 와 일치).
            #   한국 주식 일일 변동성이 ETH 의 약 1/10 → ETH 원본 MULT 20 을 200 으로 스케일 보정.
            #   이전엔 20으로 박혀 있어 service-time directionAccuracy 만 학습/검증 메트릭보다 크게 낮게 보였음.
            _GM, _ETH, _XTH = 200.0, 0.21, 0.32
            p2_arr = np.array(predictions_t2)
            p3_arr = np.array(predictions_t3)
            ec = et = xc = xt = 0
            for k in range(1, len(actual_arr) - 2):
                a_prev = actual_arr[k - 1]
                if a_prev == 0:
                    continue
                r1 = (pred_arr[k] - a_prev) / a_prev
                r2 = (p2_arr[k] - a_prev) / a_prev
                r3 = (p3_arr[k] - a_prev) / a_prev
                pr_max, pr_min = max(r1, r2, r3), min(r1, r2, r3)
                ar1 = (actual_arr[k] - a_prev) / a_prev
                ar2 = (actual_arr[k + 1] - a_prev) / a_prev
                ar3 = (actual_arr[k + 2] - a_prev) / a_prev
                ar_max, ar_min = max(ar1, ar2, ar3), min(ar1, ar2, ar3)
                if min(abs(pr_max) * _GM, 1.0) > _ETH:
                    et += 1
                    if np.sign(pr_max) == np.sign(ar_max):
                        ec += 1
                if min(abs(pr_min) * _GM, 1.0) > _XTH:
                    xt += 1
                    if np.sign(pr_min) == np.sign(ar_min):
                        xc += 1
            if et + xt > 0:
                da_e = (ec / et * 100.0) if et else None
                da_x = (xc / xt * 100.0) if xt else None
                if da_e is not None and da_x is not None:
                    direction = round((da_e + da_x) / 2.0, 2)
                else:
                    direction = round(da_e if da_e is not None else da_x, 2)

        # MODIFIED [Max/Min Trick]: 3-step 예측을 진입(max)/청산(min) 신호로 압축
        predicted_max = None
        predicted_min = None
        return_for_entry = None
        return_for_exit = None
        if (use_v5 or use_v6) and predictions_t2 and predictions_t3 and len(actuals) > 0:
            p_t1 = predictions_t1[-1]
            p_t2 = predictions_t2[-1]
            p_t3 = predictions_t3[-1]
            current = actuals[-1]
            predicted_max = round(float(max(p_t1, p_t2, p_t3)), 2)
            predicted_min = round(float(min(p_t1, p_t2, p_t3)), 2)
            if current != 0:
                return_for_entry = round(float((predicted_max - current) / current), 6)
                return_for_exit = round(float((predicted_min - current) / current), 6)

        # MODIFIED [Confidence Gating]:
        #   - 화면 표시 confidence = min(|return| * 20, 1.0) → 0~50% 범위 (예쁜 게이지 + 99% cap 회피)
        #   - 신호 결정(buy/sell signal) = |return| > **종목별 calibrated threshold** (meta 에 저장)
        # 20260602 B-option: V6 학습 시 calibrate 된 종목별 threshold 가 meta 에 있으면 그걸 사용.
        #   없으면 (V5 등 구버전) MULT=20 환경의 글로벌 default (0.0105 / 0.0160) 폴백 → 등가는
        #   confidence_entry > 0.21 / confidence_exit > 0.32 와 동일.
        GATE_MULT = 20.0  # 응답 호환 위해 유지 (실제 신뢰도 계산엔 미사용)
        _CONF_CAP = 1.0
        # 신호 결정용 raw |return| threshold — V6 calibrated 우선, 없으면 글로벌 default
        gate_entry_thr_raw = float(meta.get('gate_entry_thr_raw', 0.0105)) if isinstance(meta, dict) else 0.0105
        gate_exit_thr_raw  = float(meta.get('gate_exit_thr_raw',  0.0160)) if isinstance(meta, dict) else 0.0160
        # 응답 JSON 호환 — 기존 GATE_ENTRY_THR/EXIT_THR 필드도 같이 노출 (학습 시 calibrated 정보)
        GATE_ENTRY_THR = 0.21
        GATE_EXIT_THR  = 0.32
        confidence_entry = None
        confidence_exit = None
        buy_entry_signal = 0
        sell_exit_signal = 0
        # 자신감 = 학습 분포에서 새 예측의 percentile rank → 게이팅 컷(=1-train_ratio)과 같은 단위
        # 학습 분포가 meta 에 없는 legacy 모델은 abs(return)/gate_thr_raw 폴백
        abs_max_dist = meta.get('confidence_abs_max_dist') if isinstance(meta, dict) else None
        abs_min_dist = meta.get('confidence_abs_min_dist') if isinstance(meta, dict) else None
        if return_for_entry is not None:
            abs_ret = abs(return_for_entry)
            if abs_max_dist:
                rank = float(np.searchsorted(abs_max_dist, abs_ret)) / len(abs_max_dist)
                confidence_entry = round(min(rank, _CONF_CAP), 4)
            elif gate_entry_thr_raw > 0:
                confidence_entry = round(float(min(abs_ret / gate_entry_thr_raw, _CONF_CAP)), 4)
            else:
                confidence_entry = 1.0
            buy_entry_signal = 1 if abs_ret > gate_entry_thr_raw else 0
        if return_for_exit is not None:
            abs_ret = abs(return_for_exit)
            if abs_min_dist:
                rank = float(np.searchsorted(abs_min_dist, abs_ret)) / len(abs_min_dist)
                confidence_exit = round(min(rank, _CONF_CAP), 4)
            elif gate_exit_thr_raw > 0:
                confidence_exit = round(float(min(abs_ret / gate_exit_thr_raw, _CONF_CAP)), 4)
            else:
                confidence_exit = 1.0
            sell_exit_signal = 1 if abs_ret > gate_exit_thr_raw else 0

        # MODIFIED [Multi-Filter]: Confidence Gating 위에 RSI/BB/MACD AND 패턴
        # V5 모델만 추론 단계에서 RSI/BB/MACD를 features에서 직접 읽어 패턴 발동 여부 계산
        # 진입: rebound(과매도+하단) OR breakout(돌파+모멘텀+)
        # 청산: overheat(과매수+상단) OR breakdown(하단이탈+모멘텀-)
        rsi_now = None
        bb_pos_now = None
        macd_now = None
        pattern_rebound = 0
        pattern_breakout = 0
        pattern_overheat = 0
        pattern_breakdown = 0
        final_buy_signal = buy_entry_signal
        final_sell_signal = sell_exit_signal
        if use_v5 or use_v6:
            try:
                rsi_now = round(float(features['rsi'].iloc[-1]), 4)
                bb_pos_now = round(float(features['bb_position'].iloc[-1]), 4)
                macd_now = round(float(features['macd_hist'].iloc[-1]), 6)

                # meta에 임계값 있으면 사용, 없으면 ETH 변환 디폴트
                f_rsi_e_lo = float(meta.get('filter_rsi_entry_low', 38.0))
                f_bb_e_lo = float(meta.get('filter_bb_entry_low', 0.285))
                f_rsi_e_hi = float(meta.get('filter_rsi_entry_high', 31.0))
                f_bb_e_hi = float(meta.get('filter_bb_entry_high', 0.51))
                f_rsi_x_hi = float(meta.get('filter_rsi_exit_high', 65.0))
                f_bb_x_hi = float(meta.get('filter_bb_exit_high', 1.08))
                f_bb_x_lo = float(meta.get('filter_bb_exit_low', 0.115))

                pattern_rebound = 1 if (rsi_now < f_rsi_e_lo and bb_pos_now < f_bb_e_lo) else 0
                pattern_breakout = 1 if (rsi_now > f_rsi_e_hi and bb_pos_now > f_bb_e_hi and macd_now > 0) else 0
                pattern_overheat = 1 if (rsi_now > f_rsi_x_hi and bb_pos_now > f_bb_x_hi) else 0
                pattern_breakdown = 1 if (bb_pos_now < f_bb_x_lo and macd_now < 0) else 0

                entry_pattern_fires = (pattern_rebound == 1) or (pattern_breakout == 1)
                exit_pattern_fires = (pattern_overheat == 1) or (pattern_breakdown == 1)

                # 최종 신호 = 게이팅 통과 AND 패턴 발동
                final_buy_signal = 1 if (buy_entry_signal == 1 and entry_pattern_fires) else 0
                final_sell_signal = 1 if (sell_exit_signal == 1 and exit_pattern_fires) else 0
            except Exception as _filter_err:
                print(f"[Multi-Filter] 패턴 계산 실패 (게이팅 신호로 폴백): {_filter_err}", file=sys.stderr)

        result = {
            'code': code,
            'modelVersion': model_version,
            # MODIFIED [Model Version Routing]: 사용자 요청 vs 실제 사용 추적
            'modelRequested': requested_version_raw if requested_version_raw else 'auto',
            'modelFallback': bool(model_fallback),
            'autoTrained': bool(auto_trained),  # 20260601: 이번 예측을 위해 즉석 학습된 모델인지
            'mape': mape,
            'directionAccuracy': direction,
            'latestActual': actuals[-1],
            'latestPredicted': predictions_t1[-1],
            # MODIFIED V5: V5도 3스텝 예측이므로 분기에 포함
            'predictedT2': predictions_t2[-1] if (use_v3 or use_v4 or use_v5 or use_v6) and predictions_t2 else None,
            'predictedT3': predictions_t3[-1] if (use_v3 or use_v4 or use_v5 or use_v6) and predictions_t3 else None,
            # MODIFIED [Max/Min Trick]: 진입/청산 신호 필드
            'predictedMax': predicted_max,
            'predictedMin': predicted_min,
            'returnForEntry': return_for_entry,
            'returnForExit': return_for_exit,
            # MODIFIED [Confidence Gating]: 신뢰도 + 게이팅 시그널
            'confidenceEntry': confidence_entry,
            'confidenceExit': confidence_exit,
            'buyEntrySignal': buy_entry_signal,
            'sellExitSignal': sell_exit_signal,
            # 20260602 B-option: verify 가 같은 종목별 threshold 로 평가할 수 있도록 응답에 노출.
            'gateCalibrated': bool(isinstance(meta, dict) and meta.get('gate_calibrated', False)),
            'gateEntryThrRaw': gate_entry_thr_raw,
            'gateExitThrRaw': gate_exit_thr_raw,
            'gateEntryTrainDA': (float(meta.get('gate_entry_train_da')) if isinstance(meta, dict) and meta.get('gate_entry_train_da') is not None else None),
            'gateExitTrainDA': (float(meta.get('gate_exit_train_da')) if isinstance(meta, dict) and meta.get('gate_exit_train_da') is not None else None),
            'gateEntryTrainRatio': (float(meta.get('gate_entry_train_ratio')) if isinstance(meta, dict) and meta.get('gate_entry_train_ratio') is not None else None),
            'gateExitTrainRatio': (float(meta.get('gate_exit_train_ratio')) if isinstance(meta, dict) and meta.get('gate_exit_train_ratio') is not None else None),
            # MODIFIED [Multi-Filter]: 다중 필터 패턴 + 최종 신호 (게이팅 AND 패턴)
            'rsiNow': rsi_now,
            'bbPosNow': bb_pos_now,
            'macdNow': macd_now,
            'patternRebound': pattern_rebound,
            'patternBreakout': pattern_breakout,
            'patternOverheat': pattern_overheat,
            'patternBreakdown': pattern_breakdown,
            'finalBuySignal': final_buy_signal,
            'finalSellSignal': final_sell_signal,
            'dates': dates,
            'actuals': actuals,
            'predictions': predictions_t1,
            'predictions_t2': predictions_t2 if (use_v3 or use_v4 or use_v5 or use_v6) else None,  # MODIFIED V5
            'predictions_t3': predictions_t3 if (use_v3 or use_v4 or use_v5 or use_v6) else None   # MODIFIED V5
        }
        
        _SVC_PHASES.append(("postprocess", time.time() - _svc_postprocess_t0))
        _SVC_PHASES.append(("__overall__", time.time() - _svc_overall_t0))
        _svc_phase_report(f"{code} ({model_version}, autoTrained={auto_trained})")

        sys.stdout.reconfigure(encoding='utf-8')
        print(json.dumps(result, cls=NpEncoder))

    except Exception as e:
        print(json.dumps({"error": str(e)}, cls=NpEncoder))
        sys.exit(1)