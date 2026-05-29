package com.quant.server.controller;

import com.quant.server.dto.ApiResponse;
import com.quant.server.service.RecommendService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

/**
 * 자연어 기반 종목 추천 (AItr_manager 연동).
 * POST /api/recommend/run  body: {"query": "안정적인 대형주 3개"}
 *
 * 응답 data 는 manager 가 만든 JSON 문자열(extracted_params/scored_stocks/report/elapsed).
 * 기존 /api/backtest/predict 등과 동일하게 '원본 JSON 문자열'을 data 에 담아 전달한다.
 */
@RestController
@RequestMapping("/api/recommend")
@RequiredArgsConstructor
@Slf4j
public class RecommendController {

    private final RecommendService recommendService;

    @PostMapping("/run")
    public ResponseEntity<ApiResponse<String>> run(@RequestBody Map<String, Object> params) {
        String query = String.valueOf(params.getOrDefault("query", "")).trim();

        if (query.isEmpty()) {
            return ResponseEntity.badRequest()
                    .body(ApiResponse.error("query 가 비어 있습니다."));
        }

        log.info("[종목추천 요청] {}", query);
        String result = recommendService.runRecommend(query);

        // manager(또는 연동부)가 에러 JSON 을 반환한 경우 → 502 로 매핑
        if (result != null && result.startsWith("{\"error\"")) {
            log.warn("[종목추천] manager 에러 응답: {}", result);
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                    .body(ApiResponse.error(result));
        }

        return ResponseEntity.ok(ApiResponse.ok(result));
    }
}
