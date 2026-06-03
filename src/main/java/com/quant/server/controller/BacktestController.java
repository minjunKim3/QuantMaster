package com.quant.server.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.quant.server.domain.BacktestResult;
import com.quant.server.dto.ApiResponse;
import com.quant.server.service.BacktestResultService;
import com.quant.server.service.BacktestRunService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.jpa.repository.query.JpaEntityMetadata;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.io.File;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.LinkedHashMap;
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
        String modelVersion = (String) params.getOrDefault("modelVersion", "");
        String modelId = (String) params.getOrDefault("modelId", "");
        boolean forceTrain = Boolean.TRUE.equals(params.get("forceTrain"));

        log.info("[LSTM 예측] {} | {}일 | modelVersion={} | modelId={} | forceTrain={}",
                code, days, modelVersion.isBlank() ? "auto" : modelVersion,
                modelId.isBlank() ? "latest" : modelId, forceTrain);

        String result = backtestRunService.runLSTMPrediction(code, days, modelVersion, modelId, forceTrain);
        return ResponseEntity.ok(ApiResponse.ok(result));
    }

    // 종목/모델/기간으로 방향정확도+naive baseline 산출. modelId 명시 시 그 모델로 검증.
    @PostMapping("/verify")
    public ResponseEntity<ApiResponse<String>> verify(@RequestBody Map<String, Object> params) {
        String code = (String) params.getOrDefault("code", "KS11");
        String modelVersion = (String) params.getOrDefault("modelVersion", "V5");
        String startDate = (String) params.getOrDefault("startDate", "");
        String endDate = (String) params.getOrDefault("endDate", "");
        String modelId = (String) params.getOrDefault("modelId", "");

        log.info("[모델 검증] {} {} | modelId={} | {} ~ {}",
                code, modelVersion, modelId.isBlank() ? "(미지정)" : modelId, startDate, endDate);

        String result = backtestRunService.runVerify(code, modelVersion, startDate, endDate, modelId);
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

    // 종목별 V6 모델 존재 여부 — UI 진행률에서 "학습 vs 예측" 분기에 사용
    @GetMapping("/has-model")
    public ResponseEntity<ApiResponse<Map<String, Object>>> hasModel(@RequestParam String code) {
        String safeCode = code.replace("^", "").replace("/", "_");
        File dir = new File("agent/models_v6");
        boolean exists = false;
        if (dir.exists()) {
            File[] matches = dir.listFiles((d, n) -> n.startsWith(safeCode)
                    && (n.endsWith("_lstm_v6.pth") || n.equals(safeCode + "_lstm_v6.pth")));
            exists = matches != null && matches.length > 0;
        }
        Map<String, Object> resp = new HashMap<>();
        resp.put("code", code);
        resp.put("exists", exists);
        resp.put("version", exists ? "V6" : null);
        return ResponseEntity.ok(ApiResponse.ok(resp));
    }

    // 종목별 학습된 모델 리스트 — UI 모델 선택 박스용
    @GetMapping("/models")
    public ResponseEntity<ApiResponse<Map<String, Object>>> listModels(@RequestParam String code) {
        String safeCode = code.replace("^", "").replace("/", "_");
        File dir = new File("agent/models_v6");
        List<Map<String, Object>> models = new ArrayList<>();
        if (dir.exists()) {
            File[] files = dir.listFiles((d, n) -> n.startsWith(safeCode) && n.endsWith("_lstm_v6.pth"));
            if (files != null) {
                ObjectMapper om = new ObjectMapper();
                for (File f : files) {
                    String name = f.getName();
                    String core = name.substring(0, name.length() - "_lstm_v6.pth".length());
                    String modelId;
                    File metaJson;
                    if (core.equals(safeCode)) {
                        modelId = "legacy";
                        metaJson = null;
                    } else if (core.startsWith(safeCode + "_")) {
                        modelId = core.substring(safeCode.length() + 1);
                        metaJson = new File(dir, core + "_meta.json");
                    } else {
                        continue;
                    }
                    Map<String, Object> entry = new LinkedHashMap<>();
                    entry.put("modelId", modelId);
                    entry.put("isLegacy", "legacy".equals(modelId));
                    entry.put("fileSizeKb", f.length() / 1024);
                    if (metaJson != null && metaJson.exists()) {
                        try {
                            Map<?, ?> meta = om.readValue(metaJson, Map.class);
                            entry.put("trainedAt", meta.get("trainedAt"));
                            entry.put("gatedWeightedDaPct", meta.get("gatedWeightedDaPct"));
                            entry.put("evalDaMaxMinAvg", meta.get("evalDaMaxMinAvg"));
                            entry.put("evalDaEntryMaxMin", meta.get("evalDaEntryMaxMin"));
                            entry.put("evalDaExitMaxMin", meta.get("evalDaExitMaxMin"));
                            entry.put("trainStopReason", meta.get("trainStopReason"));
                        } catch (Exception ex) {
                            log.warn("meta.json 파싱 실패: {}", metaJson.getName(), ex);
                        }
                    } else {
                        // legacy 또는 meta.json 누락 — 파일 modtime 으로 trainedAt 대체, DA 는 null
                        entry.put("trainedAt", new Date(f.lastModified()).toInstant().toString());
                        entry.put("gatedWeightedDaPct", null);
                    }
                    models.add(entry);
                }
            }
        }
        // 최신 학습 우선
        models.sort((a, b) -> {
            String ta = String.valueOf(a.getOrDefault("trainedAt", ""));
            String tb = String.valueOf(b.getOrDefault("trainedAt", ""));
            return tb.compareTo(ta);
        });
        Map<String, Object> resp = new LinkedHashMap<>();
        resp.put("code", code);
        resp.put("models", models);
        return ResponseEntity.ok(ApiResponse.ok(resp));
    }

    // 종목/modelId 에 해당하는 모델 파일 5종 (.pth + .pkl × 3 + .json) 일괄 삭제
    @DeleteMapping("/model")
    public ResponseEntity<ApiResponse<Map<String, Object>>> deleteModel(
            @RequestParam String code, @RequestParam String modelId) {
        String safeCode = code.replace("^", "").replace("/", "_");
        File dir = new File("agent/models_v6");
        if (!dir.exists()) {
            return ResponseEntity.badRequest().body(ApiResponse.error("models_v6 디렉토리 없음"));
        }
        String prefix = "legacy".equals(modelId) ? safeCode : safeCode + "_" + modelId;
        String[] suffixes = {"_lstm_v6.pth", "_meta.pkl", "_meta.json", "_scaler.pkl", "_target_scaler.pkl"};
        List<String> deleted = new ArrayList<>();
        for (String suf : suffixes) {
            File f = new File(dir, prefix + suf);
            if (f.exists() && f.delete()) deleted.add(f.getName());
        }
        if (deleted.isEmpty()) {
            return ResponseEntity.status(404).body(ApiResponse.error(
                    "삭제할 파일 없음 (code=" + code + ", modelId=" + modelId + ")"));
        }
        log.info("[모델 삭제] {} / {} → {}개 파일 삭제", code, modelId, deleted.size());
        Map<String, Object> resp = new LinkedHashMap<>();
        resp.put("code", code);
        resp.put("modelId", modelId);
        resp.put("deletedFiles", deleted);
        return ResponseEntity.ok(ApiResponse.ok(resp));
    }

    @GetMapping("/realtime/{code}")
    public ResponseEntity<ApiResponse<String>> getRealtimeStock(
            @PathVariable String code,
            @RequestParam(defaultValue = "365") int days) {

        log.info("[실시간 데이터] {} | {}일", code, days);
        String result = backtestRunService.fetchStockData(code, days);
        return ResponseEntity.ok(ApiResponse.ok(result));
    }

    private Double toDouble(Object obj) {

        return Double.valueOf(obj.toString());
    }

    private Integer toInt(Object obj) {
        return Integer.valueOf(obj.toString());
    }

}
