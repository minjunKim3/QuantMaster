package com.quant.server.controller;

import com.quant.server.dto.ApiResponse;
import com.quant.server.service.SimulatorService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

/**
 * 과거 프랙탈 패턴 매칭 시뮬레이터.
 * POST /api/simulator/match  body: {"ticker":"005930","prediction_days":30}
 *
 * 응답 data 는 pattern_match_service.py 가 생성한 JSON 문자열.
 */
@RestController
@RequestMapping("/api/simulator")
@RequiredArgsConstructor
@Slf4j
public class SimulatorController {

    private final SimulatorService simulatorService;

    @PostMapping("/match")
    public ResponseEntity<ApiResponse<String>> match(@RequestBody Map<String, Object> params) {
        String ticker = String.valueOf(params.getOrDefault("ticker", "")).trim();
        if (ticker.isEmpty()) {
            return ResponseEntity.badRequest()
                    .body(ApiResponse.error("ticker 가 비어 있습니다."));
        }
        int predictionDays = ((Number) params.getOrDefault("prediction_days", 20)).intValue();

        log.info("[패턴매칭 요청] {} | {}일", ticker, predictionDays);

        String result = simulatorService.runMatch(ticker, predictionDays);

        if (result != null && result.startsWith("{\"error\"")) {
            log.warn("[패턴매칭] 에러 응답: {}", result);
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                    .body(ApiResponse.error(result));
        }

        return ResponseEntity.ok(ApiResponse.ok(result));
    }
}