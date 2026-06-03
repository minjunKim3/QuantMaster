package com.quant.server.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;

/**
 * AItr_simulator(과거 프랙탈 패턴 매칭) 연동 서비스.
 * BacktestRunService / RecommendService 의 ProcessBuilder + redirectErrorStream(true) 패턴 재사용.
 * V5 LSTM 과 동일한 agent/venv 사용 (fastdtw 만 추가 설치됨).
 */
@Service
@Slf4j
public class SimulatorService {

    @Value("${collector.python-path:python}")
    private String pythonPath;

    @Value("${simulator.script-path:agent/pattern_match_service.py}")
    private String scriptPath;

    @Value("${simulator.timeout-seconds:60}")
    private long timeoutSeconds;

    public String runMatch(String ticker, int predictionDays) {
        String safeTicker = ticker == null ? "" : ticker.replace("\\", "\\\\").replace("\"", "\\\"");
        String paramsJson = String.format(
                "{\"ticker\":\"%s\",\"prediction_days\":%d}", safeTicker, predictionDays);

        log.info("[패턴매칭] 파라미터: {}", paramsJson);

        Path tempFile = null;
        Process process = null;
        try {
            tempFile = Files.createTempFile("simulator_", ".json");
            Files.writeString(tempFile, paramsJson);

            ProcessBuilder pb = new ProcessBuilder(pythonPath, scriptPath, tempFile.toString());
            pb.redirectErrorStream(true);

            process = pb.start();

            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8));
            String line;
            String lastLine = "";
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("{")) {
                    lastLine = line;
                } else {
                    log.info("[simulator] {}", line);
                }
            }

            boolean finished = process.waitFor(timeoutSeconds, TimeUnit.SECONDS);
            if (!finished) {
                process.destroyForcibly();
                log.error("[패턴매칭] 타임아웃 ({}s 초과)", timeoutSeconds);
                return "{\"error\":\"패턴 매칭 시간 초과 (" + timeoutSeconds + "초)\"}";
            }

            int exitCode = process.exitValue();
            if (exitCode == 0 && !lastLine.isEmpty()) {
                log.info("[패턴매칭] 완료!");
                return lastLine;
            } else {
                log.error("[패턴매칭] 실패. 종료코드: {}", exitCode);
                return lastLine.startsWith("{")
                        ? lastLine
                        : "{\"error\":\"패턴 매칭 실행 실패 (exit " + exitCode + ")\"}";
            }
        } catch (Exception e) {
            log.error("[패턴매칭] 에러: {}", e.getMessage());
            return "{\"error\":\"" + String.valueOf(e.getMessage()).replace("\"", "'") + "\"}";
        } finally {
            if (tempFile != null) {
                try {
                    Files.deleteIfExists(tempFile);
                } catch (Exception ignore) {
                }
            }
        }
    }
}