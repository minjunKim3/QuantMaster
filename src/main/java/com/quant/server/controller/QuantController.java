package com.quant.server.controller;

import com.quant.server.domain.StockPrice;
import com.quant.server.dto.ApiResponse;
import com.quant.server.service.StockService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/stock")
@RequiredArgsConstructor
@Slf4j
public class QuantController {

    private final StockService stockService;

    @PostMapping("/save")
    public ResponseEntity<ApiResponse<StockPrice>> save(
            @RequestBody Map<String, Object> params) {
        log.info("주식 데이터 저장: {}", params.get("code"));
        StockPrice stock = stockService.save(params);
        return ResponseEntity.ok(ApiResponse.ok(stock));
    }

    @PostMapping("/save-bulk")
    public ResponseEntity<ApiResponse<String>> saveBulk(
            @RequestBody List<Map<String, Object>> paramsList) {
        log.info("주식 데이터 일괄 저장: {}건",
                paramsList.size());
        stockService.saveBulk(paramsList);
        return ResponseEntity.ok(
                ApiResponse.ok(paramsList.size() + "건 저장 완료"));
    }

    @GetMapping("/{code}")
    public ResponseEntity<ApiResponse<List<StockPrice>>> getByCode(
            @PathVariable String code,
            @RequestParam(defaultValue = "1d") String timeframe) {
        List<StockPrice> prices =
                stockService.getByCode(code, timeframe);
        return ResponseEntity.ok(ApiResponse.ok(prices));
    }

    @GetMapping("/{code}/range")
    public ResponseEntity<ApiResponse<List<StockPrice>>> getByRange(
            @PathVariable String code,
            @RequestParam String start,
            @RequestParam String end,
            @RequestParam(defaultValue = "1d") String timeframe) {
        List<StockPrice> prices =
                stockService.getByRange(code, start, end, timeframe);
        return ResponseEntity.ok(ApiResponse.ok(prices));
    }
}