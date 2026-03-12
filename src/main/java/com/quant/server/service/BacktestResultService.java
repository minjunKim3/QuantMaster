package com.quant.server.service;

import com.quant.server.domain.BacktestResult;
import com.quant.server.repository.BacktestResultRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@Service
@RequiredArgsConstructor
@Slf4j
@Transactional(readOnly = true)
public class BacktestResultService {

    private final BacktestResultRepository backtestResultRepository;

    @Transactional
    public BacktestResult save(Map<String, Object> params) {
        log.info("[디버그] 전체 params: {}", params);  // ← 이거 추가!

        BacktestResult result = new BacktestResult(
                (String) params.get("code"),
                (String) params.get("timeframe"),
                (String) params.get("strategies"),
                (String) params.get("mainModel"),
                ((Number) params.get("entryThreshold")).intValue(),
                LocalDateTime.parse((String) params.get("startDate")),
                LocalDateTime.parse((String) params.get("endDate")),
                ((Number) params.get("totalTrades")).intValue(),
                ((Number) params.get("wins")).intValue(),
                ((Number) params.get("draws")).intValue(),
                ((Number) params.get("losses")).intValue(),
                ((Number) params.get("winRate")).doubleValue(),
                ((Number) params.get("totalProfitPct")).doubleValue(),
                ((Number) params.get("maxDrawdownPct")).doubleValue()
                );
        return backtestResultRepository.save(result);
    }

    public List<BacktestResult> getByCode(String code) {
        return backtestResultRepository
                .findByCodeOrderByCreatedAtDesc(code);
    }

    public List<BacktestResult> getTopByCode(String code) {
        return backtestResultRepository
                .findByCodeOrderByTotalProfitPctDesc(code);
    }

    public List<BacktestResult> getByMainModel(String mainModel) {
        return backtestResultRepository
                .findByMainModelOrderByTotalProfitPctDesc(mainModel);
    }

}
