package com.quant.server.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;

/**
 * AItr_manager(자연어 종목 추천) 연동 서비스.
 *
 * BacktestRunService 의 ProcessBuilder + redirectErrorStream(true) 패턴을 그대로 재사용한다.
 * 단, manager 는 별도 가상환경(AItr_manager/venv)과 gguf LLM 을 쓰므로
 * V5 LSTM(BacktestRunService)과 파이썬 경로/스크립트를 분리한다.
 *
 * 주의: manager 전체 파이프라인은 약 6분(LLM 2회 + 300종목 스캔) 소요 → 타임아웃을 넉넉히 둔다.
 */
@Service
@Slf4j
public class RecommendService {

    // manager 전용 venv (V5 venv 와 분리)
    @Value("${manager.python-path:C:/QuantMaster/server/agent/AItr_manager/venv/Scripts/python.exe}")
    private String managerPythonPath;

    // manager 진입점 (CLI 1회 호출 모드 지원하도록 main.py 에 run_cli 추가됨)
    @Value("${manager.script-path:C:/QuantMaster/server/agent/AItr_manager/main.py}")
    private String managerScriptPath;

    // LLM 2회 + 종목 스캔 = ~6분. 여유 있게 기본 12분.
    @Value("${manager.timeout-seconds:720}")
    private long timeoutSeconds;

    public String runRecommend(String query) {
        // 사용자 자연어 입력만 안전하게 JSON 으로 감싸 임시파일에 기록 (lstm_service 패턴과 동일)
        String safeQuery = query == null ? "" : query.replace("\\", "\\\\").replace("\"", "\\\"");
        String paramsJson = String.format("{\"query\":\"%s\"}", safeQuery);

        log.info("[종목추천] 요청: {}", paramsJson);

        Path tempFile = null;
        Process process = null;
        try {
            tempFile = Files.createTempFile("recommend_", ".json");
            Files.writeString(tempFile, paramsJson);

            ProcessBuilder pb = new ProcessBuilder(
                    managerPythonPath, managerScriptPath, tempFile.toString());
            pb.redirectErrorStream(true);
            // manager 폴더를 작업 디렉토리로 → 상대 경로(src 패키지, gguf 등) 안전
            pb.directory(new File(managerScriptPath).getParentFile());

            process = pb.start();

            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8));
            String line;
            String lastLine = "";
            // '{' 로 시작하는 마지막 줄 = 결과 JSON. 그 외 라인은 진행 로그로 출력.
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("{")) {
                    lastLine = line;
                } else {
                    log.info("[manager] {}", line);
                }
            }

            boolean finished = process.waitFor(timeoutSeconds, TimeUnit.SECONDS);
            if (!finished) {
                process.destroyForcibly();
                log.error("[종목추천] 타임아웃 ({}s 초과)", timeoutSeconds);
                return "{\"error\":\"추천 처리 시간 초과 (" + timeoutSeconds + "초)\"}";
            }

            int exitCode = process.exitValue();

            if (exitCode == 0 && !lastLine.isEmpty()) {
                log.info("[종목추천] 완료!");
                return lastLine;
            } else {
                log.error("[종목추천] 실패. 종료코드: {}", exitCode);
                // 파이썬이 에러 JSON 을 내보냈으면 그대로 전달, 아니면 일반 메시지
                return lastLine.startsWith("{")
                        ? lastLine
                        : "{\"error\":\"종목추천 실행 실패 (exit " + exitCode + ")\"}";
            }
        } catch (Exception e) {
            log.error("[종목추천] 에러: {}", e.getMessage());
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
