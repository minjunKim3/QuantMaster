package com.quant.server.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import javax.management.StandardEmitterMBean;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

@Service
@Slf4j
public class BacktestRunService {

    @Value("${collector.python-path:python}")
    private String pythonPath;

    @Value("${backtest.script-path:C:/QuantMaster/agent/backtest_runner.py}")
    private String scriptPath;

    // NEW 20260520: 모델 성능 검증 (날짜 범위 입력 → 방향정확도 + naive baseline + train/val/test 구간)
    public String runVerify(String code, String modelVersion, String startDate, String endDate) {
        String paramsJson = String.format(
                "{\"code\":\"%s\",\"modelVersion\":\"%s\",\"startDate\":\"%s\",\"endDate\":\"%s\"}",
                code, modelVersion, startDate, endDate);

        log.info("[모델 검증] 파라미터: {}", paramsJson);

        try {
            Path tempFile = Files.createTempFile("verify_", ".json");
            Files.writeString(tempFile, paramsJson);

            String verifyScript = scriptPath.replace("backtest_runner.py", "lstm_verify.py");

            ProcessBuilder pb = new ProcessBuilder(pythonPath, verifyScript, tempFile.toString());
            pb.redirectErrorStream(true);

            Process process = pb.start();

            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()));
            String line;
            String lastLine = "";
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("{")) {
                    lastLine = line;
                } else {
                    log.info("[검증] {}", line);
                }
            }

            int exitCode = process.waitFor();
            Files.delete(tempFile);

            if (exitCode == 0 && !lastLine.isEmpty()) {
                log.info("[모델 검증] 완료!");
                return lastLine;
            } else {
                log.error("[모델 검증] 실패. 종료코드: {}", exitCode);
                return lastLine.isEmpty()
                        ? "{\"error\":\"모델 검증 실패 (exit " + exitCode + ")\"}"
                        : lastLine;
            }
        } catch (Exception e) {
            log.error("[모델 검증] 에러: {}", e.getMessage());
            return "{\"error\":\"" + e.getMessage().replace("\"", "'") + "\"}";
        }
    }

    public String runLSTMPrediction(String code, int days) {
        // 이전 호출자 호환용 오버로드 (modelVersion=auto)
        return runLSTMPrediction(code, days, "");
    }

    // NEW [Model Version Routing]: 사용자 선택 modelVersion을 파이썬으로 전달
    // modelVersion 빈 문자열 → 파이썬이 자동 폴백 (V5>V4>V3>V2)
    public String runLSTMPrediction(String code, int days, String modelVersion) {
        String safeModelVersion = modelVersion == null ? "" : modelVersion.trim();
        String paramsJson = String.format(
                "{\"code\":\"%s\",\"days\":%d,\"modelVersion\":\"%s\"}",
                code, days, safeModelVersion);

        log.info("[LSTM 예측] 파라미터: {}", paramsJson);

        try {
            Path tempFile = Files.createTempFile("lstm_", ".json");
            Files.writeString(tempFile, paramsJson);

            // lstm_service.py 경로
            String lstmScript = scriptPath.replace("backtest_runner.py", "lstm_service.py");

            ProcessBuilder pb = new ProcessBuilder(pythonPath, lstmScript, tempFile.toString());
            pb.redirectErrorStream(true);

            Process process = pb.start();

            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()));
            // stdout에서 JSON 읽기 (stderr도 합침)
            String line;
            String lastLine = "";
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("{")) {
                    lastLine = line;
                } else {
                    log.info("[LSTM] {}", line);
                }
            }

            int exitCode = process.waitFor();
            Files.delete(tempFile);

            if (exitCode == 0 && !lastLine.isEmpty()) {
                log.info("[LSTM 예측] 완료!");
                return lastLine;
            } else {
                log.error("[LSTM 예측] 실패. 종료코드: {}", exitCode);
                return "{\"error\":\"LSTM 예측 실패\"}";
            }
        } catch (Exception e) {
            log.error("[LSTM 예측] 에러: {}", e.getMessage());
            return "{\"error\":\"" + e.getMessage() + "\"}";
        }
    }

    public String runBacktest(String code, String startDate, String endDate, List<String> strategies, int threshold) {
        String strategiesJson = "[" + String.join(",", strategies.stream().map(s -> "\"" + s + "\"").toList()) + "]";
        String paramsJson = String.format(
                "{\"code\":\"%s\",\"startDate\":\"%s\",\"endDate\":\"%s\",\"strategies\":%s,\"threshold\":%d}",
                code, startDate, endDate, strategiesJson, threshold);

        log.info("[백테스트] 실행 파라미터: {}", paramsJson);

        try {
            Path tempFile = Files.createTempFile("backtest_", ".json");
            Files.writeString(tempFile, paramsJson);

            ProcessBuilder pb = new ProcessBuilder(pythonPath, scriptPath, tempFile.toString());
            pb.redirectErrorStream(true);

            Process process = pb.start();

            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));

            StringBuilder output = new StringBuilder();
            String line;
            String lastLine = "";
            while ((line = reader.readLine()) != null) {
                log.info("[Python] {}", line);
                lastLine = line;
            }

            int exitCode = process.waitFor();
            Files.delete(tempFile);

            if (exitCode == 0) {
                log.info("[백테스트] 실행 완료!");
                return lastLine;
            } else {
                log.error("[백테스트] 실패. 종료코드: {}", exitCode);
                return "{\"error\":\"백테스트 실행 실패\"}";
            }
        } catch (Exception e) {
            log.error("[백테스트] 에러: {}", e.getMessage());
            return "{\"error\":\"" + e.getMessage() + "\"}";
        }

    }

    public String runSimulation(String code, String startDate, String endDate, List<String> strategies, int threshold, int initialCash) {
        String strategiesJson = "[" + String.join(",",strategies.stream().map(s -> "\"" + s + "\"").toList()) + "]";
        String paramsJson = String.format("{\"code\":\"%s\",\"startDate\":\"%s\",\"endDate\":\"%s\",\"strategies\":%s,\"threshold\":%d,\"mode\":\"simulation\",\"initial_cash\":%d}",
                code, startDate, endDate, strategiesJson, threshold, initialCash);
        log.info("[모의투자] 실행 파라미터: {}", paramsJson);

        try {
            Path tempFile = Files.createTempFile("sim_", ".json");
            Files.writeString(tempFile, paramsJson);

            ProcessBuilder pb = new ProcessBuilder(pythonPath, scriptPath, tempFile.toString());
            pb.redirectErrorStream(true);

            Process process = pb.start();
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));

            String line;
            String lastLine = "";
            while ((line = reader.readLine()) != null) {
                log.info("[Python] {}", line);
                lastLine = line;
            }

            int exitCode = process.waitFor();
            Files.delete(tempFile);

            if (exitCode == 0) {
                log.info("[모의투자] 실행 완료!");
                return lastLine;
            } else {
                log.error("[모의투자] 실패. 종료코드: {}", exitCode);
                return "{\"error\":\"모의투자 실행 실패\"}";
            }

        } catch (Exception e) {
            log.error("[모의투자] 에러: {}", e.getMessage());
            return "{\"error\":\"" + e.getMessage() + "\"}:";
        }
    }

    public String fetchStockData(String code, int days) {
        log.info("[주식데이터] {} ({}일)", code, days);

        try {
            String stockScript = scriptPath.replace("backtest_runner.py", "stock_data.py");

            ProcessBuilder pb = new ProcessBuilder(pythonPath, stockScript, code, String.valueOf(days));
            pb.redirectErrorStream(true);

            Process process = pb.start();

            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream(), "UTF-8"));
            String line;
            String lastLine = "";
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("{")) lastLine = line;
            }

            int exitCode = process.waitFor();

            if (exitCode == 0 && !lastLine.isEmpty()) {
                return lastLine;
            } else {
                return "{\"error\":\"데이터 조회 실패\"}";
            }
        } catch (Exception e) {
            log.error("[주식데이터] 에러: {}", e.getMessage());
            return "{\"error\":\"" + e.getMessage() + "\"}";
        }
    }
}