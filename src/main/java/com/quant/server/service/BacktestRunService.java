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
}