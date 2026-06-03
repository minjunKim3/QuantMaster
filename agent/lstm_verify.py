"""
lstm_verify.py — 모델 성능 검증 도구 (V5 학습 종목 한정)

입력 JSON (sys.argv[1] 파일 경로):
    {
        "code": "KS11",
        "modelVersion": "V5",   # 표시용 (실제 폴백은 lstm_service.py가 결정)
        "startDate": "2024-01-01",
        "endDate":   "2024-12-31"
    }

출력 JSON (stdout 마지막 한 줄):
    {
        "code", "modelVersion",
        "startDate", "endDate", "samplesEvaluated",
        "directionAccuracy",   # %  (V5 게이팅 적용 시: entry/exit DA 평균)
        "avgConfidence",       # 평균 abs(예측수익률) 0~1
        "naiveBaseline",       # %  (persistence 기준)
        "improvement",         # %p (model - naive)
        "periodLabel",         # train / test / mixed  (학습이 val 미사용이라 val 라벨 제거)
        "trainStartDate", "trainEndDate", "valEndDate", "testStartDate", "fullEndDate",
        "splitBasis",          # 학습과 동일한 windowed 80/20 split 산출 근거
        # V5 Confidence Gating 메타 (V5가 아니면 false / null)
        "confidence_gating_applied",
        "samples_predicted",   # 게이팅 통과 표본 수 (entry OR exit)
        "samples_total",       # 전체 평가 후보 표본 수
        "gating_ratio",        # samples_predicted / samples_total
        "threshold_used",      # {entry, exit, mult}
        "gatedEntryAccuracy", "gatedExitAccuracy",
        "gatedEntrySamples",  "gatedExitSamples"
    }

내부 동작:
  1) FinanceDataReader 로 종목 전체 가격 시계열 로드 (2015-01-01 ~ today)
  2) 학습 코드와 동일한 windowed 80/20 split 으로 train/test 경계 날짜 산출
     (WINDOW=60, FUTURE=3, split = int((n - WINDOW - FUTURE) * 0.8); val 없음)
  3) lstm_service.py 를 subprocess 로 호출하여 모델 예측 시계열 확보
  4) 요청 [startDate, endDate] 구간만 필터
  5) V5인 경우 max/min 트릭 + Confidence Gating 적용 (lstm_train_v5.py / measure_da.py와 동일 로직)
  6) 방향정확도 / 평균신뢰도 / naive baseline / improvement 계산
"""
import json
import os
import subprocess
import sys
import tempfile
from datetime import date, datetime


# MODIFIED [Confidence Gating]: lstm_train_v5.py / measure_da.py와 동일 임계값 (한국 주식 보정용 MULT=200)
# - ETH 최적값(조원분 freqtrade) entry=0.21 / exit=0.32는 그대로
# - MULT만 ETH(20) → 한국 주식 변동성(~1/10) 보정 위해 200으로 스케일업
GATE_MULT = 200.0
GATE_ENTRY_THR = 0.21
GATE_EXIT_THR = 0.32


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _safe_print_error(msg: str) -> None:
    print(json.dumps({"error": msg}, ensure_ascii=False))


def _to_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


# MODIFIED [Split Alignment]: 학습 코드(lstm_train_v5.py:382, 396)가 train/test 80:20만 쓰고
# val을 따로 두지 않으므로 verify도 동일 의미만 노출. 라벨은 train/test/mixed 3가지.
def _period_label(start: str, end: str, train_end: str) -> str:
    try:
        s, e = _to_date(start), _to_date(end)
        t_end = _to_date(train_end)
    except Exception:
        return "unknown"
    if e <= t_end:
        return "train"   # 전 구간이 학습 데이터
    if s > t_end:
        return "test"    # 전 구간이 모델이 본 적 없는 데이터
    return "mixed"       # train/test 경계 걸침


def main() -> None:
    if len(sys.argv) < 2:
        _safe_print_error("파라미터 필요 (JSON 파일 경로)")
        sys.exit(1)

    arg = sys.argv[1]
    try:
        if arg.endswith(".json"):
            with open(arg, "r", encoding="utf-8") as f:
                params = json.load(f)
        else:
            params = json.loads(arg)
    except Exception as e:
        _safe_print_error(f"파라미터 파싱 실패: {e}")
        sys.exit(1)

    code = (params.get("code") or "KS11").strip()
    model_version = params.get("modelVersion", "V5")
    model_id_req = str(params.get("modelId", "") or "").strip()
    start_date = params.get("startDate")
    end_date = params.get("endDate")

    if not start_date or not end_date:
        _safe_print_error("startDate, endDate 필수")
        sys.exit(1)

    # 1) FDR 로 종목 전체 시계열 → train/val/test 경계 산출
    try:
        import FinanceDataReader as fdr
    except ImportError:
        _safe_print_error("FinanceDataReader 미설치")
        sys.exit(1)

    try:
        full = fdr.DataReader(code, "2015-01-01")
    except Exception as e:
        _safe_print_error(f"FDR 데이터 로드 실패: {e}")
        sys.exit(1)

    if full is None or len(full) < 100:
        _safe_print_error(f"종목 {code} 데이터 부족 ({0 if full is None else len(full)}건)")
        sys.exit(1)

    # MODIFIED [Split Alignment]: 실제 학습 split과 동일하게 계산
    # 학습 코드: X 인덱스 i ∈ [WINDOW, len(scaled) - FUTURE), split = int(len(X) * 0.8)
    # → test 시작의 full-data 인덱스 = split_idx + WINDOW
    # V3/V4/V5 모두 WINDOW=60, FUTURE=3. (V2는 FUTURE=0지만 verify 헤더에 "V5 학습 종목 한정" 명시)
    WINDOW = 60
    FUTURE = 3
    n = len(full)
    n_windowed = max(n - WINDOW - FUTURE, 0)
    split_idx = int(n_windowed * 0.8)
    test_start_full_idx = min(split_idx + WINDOW, n - 1)

    full_start = full.index[0].strftime("%Y-%m-%d")
    full_end = full.index[-1].strftime("%Y-%m-%d")
    train_end = full.index[max(test_start_full_idx - 1, 0)].strftime("%Y-%m-%d")
    test_start = full.index[test_start_full_idx].strftime("%Y-%m-%d")
    # 학습이 val을 두지 않으므로 더 이상 의미 없는 필드 — UI 호환을 위해 응답에는 null로 노출
    val_end = None

    period = _period_label(start_date, end_date, train_end)

    # 2) lstm_service.py 재호출 → 예측 시계열 확보
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lstm_script = os.path.join(script_dir, "lstm_service.py")

    try:
        end_dt = _to_date(end_date)
    except Exception:
        end_dt = date.today()
    days_needed = (date.today() - _to_date(start_date)).days + 30
    days_needed = min(max(days_needed, 120), 3000)

    tf = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    try:
        json.dump({"code": code, "days": days_needed, "modelId": model_id_req}, tf)
        tf.flush()
        tf.close()
        proc = subprocess.run(
            [sys.executable, lstm_script, tf.name],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    finally:
        try:
            os.unlink(tf.name)
        except OSError:
            pass

    last_json_line = ""
    for line in (proc.stdout or "").splitlines():
        s = line.strip()
        if s.startswith("{"):
            last_json_line = s
    if not last_json_line:
        _safe_print_error(
            "lstm_service.py 응답 없음. stderr: "
            + (proc.stderr or "")[:200].replace("\n", " ")
        )
        sys.exit(1)

    try:
        pred = json.loads(last_json_line)
    except Exception as e:
        _safe_print_error(f"lstm_service 응답 파싱 실패: {e}")
        sys.exit(1)

    if isinstance(pred, dict) and pred.get("error"):
        _safe_print_error(f"lstm_service 에러: {pred.get('error')}")
        sys.exit(1)

    dates = pred.get("dates") or []
    actuals = pred.get("actuals") or []
    predictions = pred.get("predictions") or []
    # MODIFIED [Confidence Gating]: V5 max/min 트릭에 필요한 t+2, t+3 예측 시계열도 함께 수신
    predictions_t2 = pred.get("predictions_t2") or []
    predictions_t3 = pred.get("predictions_t3") or []
    model_used = pred.get("modelVersion", model_version)

    # MODIFIED [Confidence Gating]: V5/V6 모두 게이팅 적용 (둘 다 3-step max/min 트릭 사용).
    # 20260602: V6 추가 + 종목별 calibrated threshold (B 옵션) 도입.
    gating_applicable = (
        model_used in ("V5", "V6")
        and bool(predictions_t2)
        and bool(predictions_t3)
        and len(predictions_t2) == len(predictions)
        and len(predictions_t3) == len(predictions)
    )
    # 20260602 B-option: lstm_service 응답에서 종목별 calibrated threshold 받아 사용.
    #   service 가 meta.pkl 의 gate_entry/exit_thr_raw 를 전달해줌. 없으면 글로벌 default.
    #   service 에서 사용한 MULT=20 환경의 등가값: 0.21/20=0.0105, 0.32/20=0.0160.
    gate_entry_thr_raw = float(pred.get("gateEntryThrRaw", 0.0105))
    gate_exit_thr_raw  = float(pred.get("gateExitThrRaw",  0.0160))
    gate_calibrated   = bool(pred.get("gateCalibrated", False))

    # 3) 요청 구간 필터 (V5면 t1/t2/t3 quintuple, 아니면 t1만)
    quintuples = []  # (date, actual, p_t1, p_t2, p_t3 or None)
    for i, d in enumerate(dates):
        if not (isinstance(d, str) and len(d) >= 10):
            continue
        d10 = d[:10]
        if d10 < start_date or d10 > end_date:
            continue
        if i >= len(actuals) or i >= len(predictions):
            continue
        a = actuals[i]
        p = predictions[i]
        if a is None or p is None:
            continue
        p2 = None
        p3 = None
        if gating_applicable and i < len(predictions_t2) and i < len(predictions_t3):
            p2v = predictions_t2[i]
            p3v = predictions_t3[i]
            if p2v is not None and p3v is not None:
                p2 = float(p2v)
                p3 = float(p3v)
        quintuples.append((d10, float(a), float(p), p2, p3))

    if len(quintuples) < 3:
        _safe_print_error(
            f"선택 구간의 예측 샘플이 부족합니다 ({len(quintuples)}개). "
            f"가능한 모델 시계열: {dates[0][:10] if dates else '-'} ~ {dates[-1][:10] if dates else '-'}"
        )
        sys.exit(1)

    # 4) 방향정확도 + 평균 신뢰도 + Confidence Gating (V5 전용)
    # ────────────────────────────────────────────────────────────────
    # MODIFIED [Confidence Gating]: lstm_train_v5.py / measure_da.py와 동일 로직 이식
    #   - 각 표본의 (t+1, t+2, t+3) 예측을 prev_actual 기준 수익률로 환산
    #   - max/min 트릭: pred_ret_max(entry 신호), pred_ret_min(exit 신호)
    #   - conf = min(|return| * MULT, 1.0) → threshold 미달은 신호 X
    #   - 통과 표본만 entry/exit DA 계산, 평균이 directionAccuracy
    # 비-V5 모델은 기존 단일 step DA 그대로 유지 (게이팅 비활성)
    # ────────────────────────────────────────────────────────────────
    legacy_correct = 0
    legacy_total = 0
    conf_list = []

    # 게이팅 카운터
    entry_correct = 0
    entry_total = 0
    exit_correct = 0
    exit_total = 0
    # 신호 발생률 = (entry 통과 + exit 통과) / (samples_total × 2)
    # OR 조건이었던 이전 방식은 entry 100% 통과 종목에선 무조건 100% 표시되어 게이팅이 가려졌음
    entry_pass_count = 0
    exit_pass_count = 0
    samples_total = 0      # 게이팅 평가 후보 표본 수 (미래 actual 충분히 있는 것)

    for i in range(1, len(quintuples)):
        _, a_prev, _, _, _ = quintuples[i - 1]
        _, a_cur, p_cur, p2_cur, p3_cur = quintuples[i]

        # 단일 step 평균 신뢰도 (기존 의미 유지)
        if a_prev:
            conf_list.append(min(abs(p_cur - a_prev) / a_prev, 1.0))

        # legacy DA — 게이팅 OFF 시 단일 step 정확도 (모든 V5 표본도 포함)
        legacy_actual_dir = _sign(a_cur - a_prev)
        legacy_pred_dir = _sign(p_cur - a_prev)
        if legacy_actual_dir != 0:
            legacy_total += 1
            if legacy_actual_dir == legacy_pred_dir:
                legacy_correct += 1

        # 비-V5 또는 t2/t3 결측: 게이팅 적용 불가 → 위에서 legacy만 카운트하고 종료
        if not gating_applicable or p2_cur is None or p3_cur is None:
            continue

        # V5 + 게이팅 적용 경로
        if not a_prev:
            continue

        # 예측 수익률 (lstm_service에서 이미 systematic_bias 반영된 가격 → 수익률 환산)
        ret_t1 = (p_cur - a_prev) / a_prev
        ret_t2 = (p2_cur - a_prev) / a_prev
        ret_t3 = (p3_cur - a_prev) / a_prev
        pred_ret_max = max(ret_t1, ret_t2, ret_t3)
        pred_ret_min = min(ret_t1, ret_t2, ret_t3)

        # actual t+1/t+2/t+3 — quintuples는 거래일 연속이므로 i, i+1, i+2 인덱스로 접근
        # 미래 actual이 부족하면 (마지막 ~2개 표본) 게이팅 평가 불가 → 카운트 제외
        if i + 2 >= len(quintuples):
            continue
        a_t1 = a_cur
        a_t2 = float(quintuples[i + 1][1])
        a_t3 = float(quintuples[i + 2][1])
        actual_ret_t1 = (a_t1 - a_prev) / a_prev
        actual_ret_t2 = (a_t2 - a_prev) / a_prev
        actual_ret_t3 = (a_t3 - a_prev) / a_prev
        actual_ret_max = max(actual_ret_t1, actual_ret_t2, actual_ret_t3)
        actual_ret_min = min(actual_ret_t1, actual_ret_t2, actual_ret_t3)

        # 신뢰도 (화면 표시 호환) + 게이팅 마스크 (실제 결정은 raw |return| vs 종목별 threshold)
        conf_entry = min(abs(pred_ret_max) * GATE_MULT, 1.0)
        conf_exit = min(abs(pred_ret_min) * GATE_MULT, 1.0)
        # 20260602 B-option: raw |return| 기준으로 게이팅 — service 와 동일 로직
        entry_pass = abs(pred_ret_max) > gate_entry_thr_raw
        exit_pass  = abs(pred_ret_min) > gate_exit_thr_raw

        samples_total += 1
        if entry_pass:
            entry_pass_count += 1
            entry_total += 1
            if _sign(pred_ret_max) == _sign(actual_ret_max):
                entry_correct += 1
        if exit_pass:
            exit_pass_count += 1
            exit_total += 1
            if _sign(pred_ret_min) == _sign(actual_ret_min):
                exit_correct += 1
    samples_predicted = entry_pass_count + exit_pass_count

    # directionAccuracy 는 통과 표본 수로 가중평균 — 학습 시 gatedWeightedDaPct 와 같은 공식
    legacy_direction_acc = round((legacy_correct / legacy_total) * 100.0, 2) if legacy_total else 0.0
    if gating_applicable and (entry_total + exit_total) > 0:
        da_entry = (entry_correct / entry_total) * 100.0 if entry_total else None
        da_exit = (exit_correct / exit_total) * 100.0 if exit_total else None
        if da_entry is not None and da_exit is not None:
            direction_acc = round(
                (entry_total * da_entry + exit_total * da_exit) / (entry_total + exit_total), 2
            )
        elif da_entry is not None:
            direction_acc = round(da_entry, 2)
        else:
            direction_acc = round(da_exit, 2)
        samples_evaluated = samples_predicted
    else:
        direction_acc = legacy_direction_acc
        samples_evaluated = legacy_total
        da_entry = None
        da_exit = None

    avg_conf = round(sum(conf_list) / len(conf_list), 4) if conf_list else 0.0

    # 5) naive baseline (persistence: yesterday's direction → today's prediction)
    n_correct = 0
    n_total = 0
    for i in range(2, len(quintuples)):
        _, a_pp, _, _, _ = quintuples[i - 2]
        _, a_p, _, _, _ = quintuples[i - 1]
        _, a_c, _, _, _ = quintuples[i]
        actual_dir = _sign(a_c - a_p)
        naive_dir = _sign(a_p - a_pp)
        if actual_dir != 0:
            n_total += 1
            if naive_dir == actual_dir:
                n_correct += 1
    naive_acc = round((n_correct / n_total) * 100.0, 2) if n_total else 0.0
    improvement = round(direction_acc - naive_acc, 2)

    # MODIFIED [Confidence Gating]: 게이팅 메타 필드
    # 분모 ×2: entry/exit 각각 평가했으므로 표본당 2번 카운트 가능
    gating_ratio = (
        round(samples_predicted / (samples_total * 2), 4) if samples_total else 0.0
    )

    result = {
        "code": code,
        "modelVersion": model_used,
        "modelRequested": model_version,
        "startDate": start_date,
        "endDate": end_date,
        "samplesEvaluated": samples_evaluated,
        "directionAccuracy": direction_acc,
        "avgConfidence": avg_conf,
        "naiveBaseline": naive_acc,
        "improvement": improvement,
        "periodLabel": period,
        "trainStartDate": full_start,
        "trainEndDate": train_end,
        "valEndDate": val_end,           # 학습 split에 val 없음 → 항상 null
        "testStartDate": test_start,
        "fullEndDate": full_end,
        # MODIFIED [Split Alignment]: split 산출 근거 메타 (학술적 신뢰성)
        "splitBasis": f"windowed 80/20 (WINDOW={WINDOW}, FUTURE={FUTURE}, n_windowed={n_windowed}, split_idx={split_idx})",
        # MODIFIED [Confidence Gating]: V5 게이팅 메타
        "confidence_gating_applied": bool(gating_applicable),
        "samples_predicted": samples_predicted if gating_applicable else None,
        "samples_total": samples_total if gating_applicable else None,
        "gating_ratio": gating_ratio if gating_applicable else None,
        "threshold_used": {
            "entry": GATE_ENTRY_THR,
            "exit": GATE_EXIT_THR,
            "mult": GATE_MULT,
            # 20260602 B-option: 종목별 calibrated threshold (실제 게이팅 기준)
            "entry_raw": gate_entry_thr_raw,
            "exit_raw": gate_exit_thr_raw,
            "calibrated": gate_calibrated,
        } if gating_applicable else None,
        "gatedEntryAccuracy": round(da_entry, 2) if (gating_applicable and da_entry is not None) else None,
        "gatedExitAccuracy": round(da_exit, 2) if (gating_applicable and da_exit is not None) else None,
        "gatedEntrySamples": entry_total if gating_applicable else None,
        "gatedExitSamples": exit_total if gating_applicable else None,
        # 비교용 디버그: 게이팅 OFF 시 단일 step DA (모든 표본 강제 평가, 이식 전과 동일)
        "legacyDirectionAccuracy": legacy_direction_acc,
        "legacySamplesEvaluated": legacy_total,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
