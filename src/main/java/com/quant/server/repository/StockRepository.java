package com.quant.server.repository;

import com.quant.server.domain.StockPrice;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;

@Repository
public interface StockRepository extends JpaRepository<StockPrice, Long>{

    List<StockPrice> findByCodeAndTimeframeOrderByTradeTimeAsc(
            String code, String timeframe);

    List<StockPrice> findByCodeAndTimeframeAndTradeTimeBetweenOrderByTradeTimeAsc(
            String code, String timeframe,
            LocalDateTime start, LocalDateTime end);

    StockPrice findTopByCodeAndTimeframeOrderByTradeTimeDesc(
            String code, String timeframe);
}
