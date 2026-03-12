package com.quant.server.service;

import com.quant.server.domain.StockPrice;
import com.quant.server.repository.StockRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class StockService {

    private final StockRepository stockRepository;

    @Transactional
    public StockPrice save(Map<String, Object> params) {
        StockPrice stock = new StockPrice(
                (String) params.get("code"),
                (Double) params.get("open"),
                (Double) params.get("high"),
                (Double) params.get("low"),
                (Double) params.get("close"),
                ((Number) params.get("volume")).longValue(),
                (String) params.get("timeframe"),
                LocalDateTime.parse((String) params.get("tradeTime"))
                );
        return stockRepository.save(stock);
    }

    @Transactional
    public void saveBulk(List<Map<String, Object>> paramsList) {
        for (Map<String, Object> params : paramsList) {
            save(params);
        }
    }

    public List<StockPrice> getByCode(String code, String timeframe) {
        return stockRepository
                .findByCodeAndTimeframeOrderByTradeTimeAsc(code, timeframe);
    }

    public List<StockPrice> getByRange(
            String code, String timeframe,
            String start, String end) {
        LocalDateTime startDate = LocalDateTime.parse(start);
        LocalDateTime endDate = LocalDateTime.parse(end);
        return stockRepository
                .findByCodeAndTimeframeAndTradeTimeBetweenOrderByTradeTimeAsc(code, timeframe, startDate, endDate);
        }

    public LocalDateTime getLatestTradeTime(String code, String timeframe) {
        StockPrice latest = stockRepository
                .findTopByCodeAndTimeframeOrderByTradeTimeDesc(code, timeframe);
        return latest != null ? latest.getTradeTime() : null;
    }
}
