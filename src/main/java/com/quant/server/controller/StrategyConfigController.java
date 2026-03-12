package com.quant.server.controller;

import com.quant.server.domain.StrategyConfig;
import com.quant.server.dto.ApiResponse;
import com.quant.server.service.StrategyConfigService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;
import java.util.List;

@RestController
@RequestMapping("/api/strategy")
@RequiredArgsConstructor
@Slf4j
public class StrategyConfigController {
    private final StrategyConfigService strategyConfigService;

    @PostMapping
    public ResponseEntity<ApiResponse<StrategyConfig>> create(
            @RequestBody Map<String, Object> params) {
        log.info("전략 설정 생성: {}",
                params.get("name"));
        StrategyConfig config = strategyConfigService.create(params);
        return ResponseEntity.ok(ApiResponse.ok(config));
    }

    @GetMapping
    public ResponseEntity<ApiResponse<List<StrategyConfig>>> getAll() {
        List<StrategyConfig> list = strategyConfigService.getAll();
        return ResponseEntity.ok(ApiResponse.ok(list));
    }

    @GetMapping("/{id}")
    public ResponseEntity<ApiResponse<StrategyConfig>> getById(@PathVariable Long id) {
        StrategyConfig config = strategyConfigService.getById(id);
        return ResponseEntity.ok(ApiResponse.ok(config));
    }

    @GetMapping("/active")
    public ResponseEntity<ApiResponse<StrategyConfig>> getActive() {
        StrategyConfig config = strategyConfigService.getActive();
        return ResponseEntity.ok(ApiResponse.ok(config));
    }

    @PutMapping("/{id}")
    public ResponseEntity<ApiResponse<StrategyConfig>> update(
            @PathVariable Long id,
            @RequestBody Map<String, Object> params) {

        log.info("전략 설정 수정 - id: [}", id);

        StrategyConfig config = strategyConfigService.update(id, params);

        return ResponseEntity.ok(ApiResponse.ok(config));
    }

    @PutMapping("/{id}/activate")
    public ResponseEntity<ApiResponse<StrategyConfig>> activate(@PathVariable Long id) {
        log.info("전략 활성화 - id: {}", id);
        StrategyConfig config = strategyConfigService.activate(id);
        return ResponseEntity.ok(ApiResponse.ok(config));
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<ApiResponse<String>> delete(
            @PathVariable Long id) {
        strategyConfigService.delete(id);
        return ResponseEntity.ok(ApiResponse.ok("삭제 완료! id=" + id));
    }
}
