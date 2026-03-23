package com.quant.server.controller;

import com.quant.server.domain.BacktestResult;
import com.quant.server.dto.ApiResponse;
import com.quant.server.service.BacktestResultService;
import com.quant.server.service.BacktestRunService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.jpa.repository.query.JpaEntityMetadata;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/backtest")
@RequiredArgsConstructor
@Slf4j
public class BacktestController {

    private final BacktestResultService backtestResultService;
    private final BacktestRunService backtestRunService;

    @PostMapping("/save")
    public ResponseEntity<ApiResponse<BacktestResult>> save(
            @RequestBody Map<String, Object> params) {

        log.info("백테스트 결과 수신: {}", params.get("strategies"));

        BacktestResult result = backtestResultService.save(params);
        return ResponseEntity.ok(ApiResponse.ok(result));
    }

    @GetMapping("/{code}")
    public ResponseEntity<ApiResponse<List<BacktestResult>>> getByCode(
            @PathVariable String code) {
        List<BacktestResult> results = backtestResultService.getByCode(code);
        return ResponseEntity.ok(ApiResponse.ok(results));
    }

    @GetMapping("/{code}/top")
    public ResponseEntity<ApiResponse<List<BacktestResult>>> getTopByCode(
            @PathVariable String code) {
        List<BacktestResult> results = backtestResultService.getTopByCode(code);
        return ResponseEntity.ok(ApiResponse.ok(results));
    }

    @GetMapping("/model/{model}")
    public ResponseEntity<ApiResponse<List<BacktestResult>>> getByMainModel(
            @PathVariable String model) {
        List<BacktestResult> results = backtestResultService.getByMainModel(model);
        return ResponseEntity.ok(ApiResponse.ok(results));
    }

    @PostMapping("/run")
    public ResponseEntity<ApiResponse<String>> run(@RequestBody Map<String, Object> params) {
        String code = (String) params.get("code");
        String startDate = (String) params.get("startDate");
        String endDate = (String) params.get("endDate");
        List<String> strategies = (List<String>) params.get("strategies");
        int threshold = (Integer) params.get("threshold");

        log.info("[백테스트 실행] {} | 전략: {} | 기간: {}~{}",
                code, strategies, startDate, endDate);

        String result = backtestRunService.runBacktest(code, startDate, endDate, strategies, threshold);

        return ResponseEntity.ok(ApiResponse.ok(result));
    }

    @PostMapping("/simulation")
    public ResponseEntity<ApiResponse<String>> simulation(@RequestBody Map<String, Object> params) {
        String code = (String) params.get("code");
        String startDate = (String) params.get("startDate");
        String endDate = (String) params.get("endDate");
        List<String> strategies = (List<String>) params.get("strategies");
        int threshold = ((Number) params.get("threshold")).intValue();
        int initialCash = ((Number) params.get("initialCash")).intValue();

        log.info("[모의투자] {} | 전략: {} | 자금: ${}", code, strategies, initialCash);

        String result = backtestRunService.runSimulation(code, startDate, endDate, strategies, threshold, initialCash);

        return ResponseEntity.ok(ApiResponse.ok(result));
    }

    @PostMapping("/predict")
    public ResponseEntity<ApiResponse<String>> predict(@RequestBody Map<String, Object> params) {
        String code = (String) params.getOrDefault("code", "KS11");
        int days = ((Number) params.getOrDefault("days", 100)).intValue();

        log.info("[LSTM 예측] {} | 최근 {}일", code, days);

        String result = backtestRunService.runLSTMPrediction(code, days);
        return ResponseEntity.ok(ApiResponse.ok(result));
    }

    @PostMapping("/agent")
    public ResponseEntity<ApiResponse<String>> agentAnalysis(@RequestBody Map<String, Object> params) {
        String query = (String) params.getOrDefault("query", "");
        int days = ((Number) params.getOrDefault("days", 7)).intValue();

        log.info("[AI Agent] {} | {}일", query, days);

        String result = backtestRunService.runAgentAnalysis(query, days);
        return ResponseEntity.ok(ApiResponse.ok(result));
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<ApiResponse<String>> deleteBacktest(@PathVariable Long id) {
        backtestResultService.deleteById(id);
        return ResponseEntity.ok(ApiResponse.ok("삭제 완료"));
    }

    @DeleteMapping("/clear")
    public ResponseEntity<ApiResponse<String>> clearBacktests() {
        backtestResultService.deleteAll();
        return ResponseEntity.ok(ApiResponse.ok("전체 삭제 완료"));
    }


    private Double toDouble(Object obj) {

        return Double.valueOf(obj.toString());
    }

    private Integer toInt(Object obj) {
        return Integer.valueOf(obj.toString());
    }

}
