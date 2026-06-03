package com.quant.server.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Python 에이전트(backtest_runner / lstm_service / lstm_verify / stock_data) 호출 래퍼.
 *
 * 20260602 리팩토링:
 *  - 5개 메서드가 동일했던 [params 직렬화 → 임시파일 → ProcessBuilder → stdout 마지막 `{` 라인 추출 → 정리]
 *    패턴을 {@link #runPythonAndExtractJson} / {@link #runPythonWithJsonParam} 두 헬퍼로 통합.
 *  - 모든 stdout 리더를 UTF-8 명시 (Windows cp949 디폴트 회피).
 *  - 임시파일 정리를 finally 블록의 deleteIfExists 로 통일 (이전엔 일부 throw 가능 경로).
 */
@Service
@Slf4j
public class BacktestRunService {

    @Value("${collector.python-path:python}")
    private String pythonPath;

    @Value("${backtest.script-path:agent/backtest_runner.py}")
    private String scriptPath;

    // 20260602: V5(약 195초) → V6(B + LSTM time-budget 120s, 보통 약 2~3분)으로 단축.
    //  안전망 5분(300초) — 데이터 download / Chronos 첫 weight 캐시 등 외부 지연 흡수.
    private static final int LSTM_TIMEOUT_SECONDS = 300;
    private static final int NO_TIMEOUT = 0;

    // 모델 성능 검증 (날짜 범위 입력 → 방향정확도 + naive baseline + train/val/test 구간)
    public String runVerify(String code, String modelVersion, String startDate, String endDate) {
        return runVerify(code, modelVersion, startDate, endDate, "");
    }

    // modelId 는 lstm_verify.py → lstm_service.py 까지 그대로 전달되어 해당 모델로 검증
    public String runVerify(String code, String modelVersion, String startDate, String endDate, String modelId) {
        String safeModelId = modelId == null ? "" : modelId.trim();
        String paramsJson = String.format(
                "{\"code\":\"%s\",\"modelVersion\":\"%s\",\"startDate\":\"%s\",\"endDate\":\"%s\",\"modelId\":\"%s\"}",
                code, modelVersion, startDate, endDate, safeModelId);
        return runPythonWithJsonParam("lstm_verify.py", paramsJson, NO_TIMEOUT, "모델 검증");
    }

    public String runLSTMPrediction(String code, int days) {
        return runLSTMPrediction(code, days, "", "", false);
    }

    public String runLSTMPrediction(String code, int days, String modelVersion) {
        return runLSTMPrediction(code, days, modelVersion, "", false);
    }

    // modelId 명시 → 그 모델 / 빈 문자열 → 최신 timestamp → legacy 폴백
    // forceTrain=true → modelId 무시 + 새 학습
    public String runLSTMPrediction(String code, int days, String modelVersion,
                                    String modelId, boolean forceTrain) {
        String safeModelVersion = modelVersion == null ? "" : modelVersion.trim();
        String safeModelId = modelId == null ? "" : modelId.trim();
        String paramsJson = String.format(
                "{\"code\":\"%s\",\"days\":%d,\"modelVersion\":\"%s\",\"modelId\":\"%s\",\"forceTrain\":%s}",
                code, days, safeModelVersion, safeModelId, forceTrain);
        return runPythonWithJsonParam("lstm_service.py", paramsJson, LSTM_TIMEOUT_SECONDS, "LSTM 예측");
    }

    public String runBacktest(String code, String startDate, String endDate, List<String> strategies, int threshold) {
        String strategiesJson = "[" + String.join(",", strategies.stream().map(s -> "\"" + s + "\"").toList()) + "]";
        String paramsJson = String.format(
                "{\"code\":\"%s\",\"startDate\":\"%s\",\"endDate\":\"%s\",\"strategies\":%s,\"threshold\":%d}",
                code, startDate, endDate, strategiesJson, threshold);
        // scriptPath 가 backtest_runner.py 그 자체이므로 헬퍼에 그대로 위임.
        return runPythonWithJsonParam("backtest_runner.py", paramsJson, NO_TIMEOUT, "백테스트");
    }

    public String runSimulation(String code, String startDate, String endDate, List<String> strategies, int threshold, int initialCash) {
        String strategiesJson = "[" + String.join(",", strategies.stream().map(s -> "\"" + s + "\"").toList()) + "]";
        String paramsJson = String.format(
                "{\"code\":\"%s\",\"startDate\":\"%s\",\"endDate\":\"%s\",\"strategies\":%s,\"threshold\":%d,\"mode\":\"simulation\",\"initial_cash\":%d}",
                code, startDate, endDate, strategiesJson, threshold, initialCash);
        return runPythonWithJsonParam("backtest_runner.py", paramsJson, NO_TIMEOUT, "모의투자");
    }

    public String fetchStockData(String code, int days) {
        log.info("[주식데이터] {} ({}일)", code, days);
        // stock_data.py 는 JSON 임시파일 대신 위치 인자(code, days) 를 받음 — 헬퍼에 직접 위임.
        return runPythonAndExtractJson(
                "stock_data.py",
                List.of(code, String.valueOf(days)),
                NO_TIMEOUT,
                "주식데이터"
        );
    }

    // ---------------------------------------------------------------------
    // 헬퍼: JSON params 를 임시파일로 떨궈 호출하는 케이스
    // ---------------------------------------------------------------------
    private String runPythonWithJsonParam(String scriptName, String paramsJson, int timeoutSec, String logPrefix) {
        log.info("[{}] 파라미터: {}", logPrefix, paramsJson);
        Path tempFile = null;
        try {
            tempFile = Files.createTempFile(scriptName.replace(".py", "_"), ".json");
            Files.writeString(tempFile, paramsJson);
            return runPythonAndExtractJson(scriptName, List.of(tempFile.toString()), timeoutSec, logPrefix);
        } catch (Exception e) {
            log.error("[{}] 에러: {}", logPrefix, e.getMessage());
            return "{\"error\":\"" + e.getMessage().replace("\"", "'") + "\"}";
        } finally {
            if (tempFile != null) {
                try {
                    Files.deleteIfExists(tempFile);
                } catch (Exception ignored) {
                    // 임시파일 삭제 실패는 무시 (OS 가 결국 정리)
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // 헬퍼: ProcessBuilder + stdout 마지막 JSON 라인 추출 공통 로직
    //  - stderr 는 redirectErrorStream(true) 로 stdout 에 합침.
    //  - JSON 결과는 항상 `{` 로 시작하는 라인. 그 외 라인은 INFO 로 로깅.
    //  - timeoutSec = NO_TIMEOUT(0) 이면 무한 대기 (백테스트/모의투자), 양수면 그만큼 후 destroyForcibly.
    // ---------------------------------------------------------------------
    private String runPythonAndExtractJson(String scriptName, List<String> extraArgs, int timeoutSec, String logPrefix) {
        try {
            // backtest_runner.py 절대경로 → 같은 디렉터리 안의 scriptName 으로 변환.
            String script = scriptPath.endsWith(scriptName)
                    ? scriptPath
                    : scriptPath.replace("backtest_runner.py", scriptName);

            ProcessBuilder pb = new ProcessBuilder();
            pb.command().add(pythonPath);
            pb.command().add(script);
            pb.command().addAll(extraArgs);
            pb.redirectErrorStream(true);

            Process process = pb.start();

            String lastLine = "";
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.startsWith("{")) {
                        lastLine = line;
                    } else {
                        log.info("[{}] {}", logPrefix, line);
                    }
                }
            }

            int exitCode;
            if (timeoutSec > 0) {
                if (!process.waitFor(timeoutSec, TimeUnit.SECONDS)) {
                    process.destroyForcibly();
                    log.error("[{}] 타임아웃 ({}초 초과)", logPrefix, timeoutSec);
                    return "{\"error\":\"" + logPrefix + " 시간 초과 (" + timeoutSec + "초)\"}";
                }
                exitCode = process.exitValue();
            } else {
                exitCode = process.waitFor();
            }

            if (exitCode == 0 && !lastLine.isEmpty()) {
                log.info("[{}] 완료", logPrefix);
                return lastLine;
            }

            log.error("[{}] 실패. 종료코드: {}", logPrefix, exitCode);
            // 파이썬이 에러 JSON 을 내보냈으면 그대로 전달 (자동 학습 실패 메시지 등 노출).
            return lastLine.startsWith("{")
                    ? lastLine
                    : "{\"error\":\"" + logPrefix + " 실패 (exit " + exitCode + ")\"}";
        } catch (Exception e) {
            log.error("[{}] 에러: {}", logPrefix, e.getMessage());
            String msg = e.getMessage() == null ? "" : e.getMessage().replace("\"", "'");
            return "{\"error\":\"" + msg + "\"}";
        }
    }
}