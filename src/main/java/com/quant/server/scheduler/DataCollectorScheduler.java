package com.quant.server.scheduler;

import jakarta.persistence.criteria.CriteriaBuilder;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.time.LocalDateTime;

@Component
@Slf4j
public class DataCollectorScheduler {

    @Value("${collector.python-path:python}")
    private String pythonPath;

    @Value("${collector.script-path:C:/QuantMaster/python/stock_collector.py}")
    private String scriptPath;

    @Value("${collector.enabled:false}")
    private boolean enabled;

    @Scheduled(cron = "0 0 9 * * *")
    public void collectDailyData() {
        if (!enabled) {
            log.info("[스케쥴러] 비활성 상태 - 수집 건너뜀");
            return;
        }

        log.info("[스케쥴러] 일일 데이터 수집 시작 - {}", LocalDateTime.now());

        try {
            ProcessBuilder pb = new ProcessBuilder(pythonPath, scriptPath);
            pb.redirectErrorStream(true);

            Process process = pb.start();

            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));

            String line;
            while ((line = reader.readLine()) != null) {
                log.info("[Python] {}", line);
            }

            int exitCode = process.waitFor();

            if (exitCode == 0) {
                log.info("[스케쥴러] 데이터 수집 완료!");
            } else {
                log.error("[스케줄러] Python 스크립트 실패. 종료코드: {}", exitCode);
            }
        } catch (Exception e) {
            log.error("[스케쥴러] 데이터 수집 중 에러: {}", e.getMessage());
        }
    }

    @Scheduled(fixedRate = 30000)
    private void healthCheck() {
        log.info("[스케쥴러] 상태 체크 - 활성: {}, 시각: {}", enabled, LocalDateTime.now());
    }
}
