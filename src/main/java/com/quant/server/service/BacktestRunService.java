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

    public String runLSTMPrediction(String code, int days) {
        String paramsJson = String.format(
                "{\"code\":\"%s\",\"days\":%d}",
                code, days);

        log.info("[LSTM 예측] 파라미터: {}", paramsJson);

        try {
            Path tempFile = Files.createTempFile("lstm_", ".json");
            Files.writeString(tempFile, paramsJson);

            // lstm_service.py 경로
            String lstmScript = scriptPath.replace("backtest_runner.py", "lstm_service.py");

            ProcessBuilder pb = new ProcessBuilder(pythonPath, lstmScript, tempFile.toString());
            pb.redirectErrorStream(false);

            Process process = pb.start();

            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()));
            BufferedReader errReader = new BufferedReader(
                    new InputStreamReader(process.getErrorStream()));

            // stderr는 로그로
            String errLine;
            while ((errLine = errReader.readLine()) != null) {
                log.info("[LSTM] {}", errLine);
            }

            // stdout에서 JSON 읽기
            String line;
            String lastLine = "";
            while ((line = reader.readLine()) != null) {
                lastLine = line;
            }

            int exitCode = process.waitFor();
            Files.delete(tempFile);

            if (exitCode == 0) {
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

    public String runAgentAnalysis(String query, int days) {
        String paramsJson = String.format(
                "{\"query\":\"%s\",\"days\":%d}",
                query, days);

        log.info("[AI Agent] 분석 요청: {}", query);

        try {
            Path tempFile = Files.createTempFile("agent_", ".json");
            Files.writeString(tempFile, paramsJson);

            String agentScript = scriptPath.replace("backtest_runner.py", "agent_analysis.py");

            ProcessBuilder pb = new ProcessBuilder(pythonPath, agentScript, tempFile.toString());
            pb.redirectErrorStream(true);

            Process process = pb.start();

            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream(), "UTF-8"));
            String line;
            String lastLine = "";
            while ((line = reader.readLine()) != null) {
                lastLine = line;
            }

            int exitCode = process.waitFor();
            Files.delete(tempFile);

            if (exitCode == 0) {
                log.info("[AI Agent] 분석 완료!");
                return lastLine;
            } else {
                return "{\"error\":\"AI Agent 분석 실패\"}";
            }
        } catch (Exception e) {
            log.error("[AI Agent] 에러: {}", e.getMessage());
            return "{\"error\":\"" + e.getMessage() + "\"}";
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