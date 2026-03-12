package com.quant.server.domain;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.hibernate.boot.jaxb.internal.stax.LocalSchemaLocator;
import org.springframework.cglib.core.Local;

import java.time.LocalDateTime;

@Entity
@Table(name = "backtest_result")
@Getter
@NoArgsConstructor
public class BacktestResult {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private long id;

    @Column(nullable = false, length = 20)
    private String code;

    @Column(nullable = false, length = 10)
    private String timeframe;

    @Column(nullable = false)
    private String strategies;

    @Column(name = "main_model", length = 20)
    private String mainModel;

    @Column(name = "entry_threshold")
    private Integer entryThreshold;

    @Column(name = "start_date", nullable = false)
    private LocalDateTime startDate;

    @Column(name = "end_date", nullable = false)
    private LocalDateTime endDate;

    @Column(name = "total_trades")
    private Integer totalTrades;

    private Integer wins;
    private Integer draws;
    private Integer losses;

    @Column(name = "win_rate")
    private Double winRate;

    @Column(name = "total_profit_pct")
    private Double totalProfitPct;

    @Column(name = "max_drawdown_pct")
    private Double maxDrawdownPct;

    @Column(name = "created_at", nullable = false)
    private LocalDateTime createdAt;

    public BacktestResult(String code, String timeframe, String strategies, String mainModel,
                          Integer entryThreshold, LocalDateTime startDate, LocalDateTime endDate,
                          Integer totalTrades, Integer wins, Integer draws, Integer losses, Double winRate,
                          Double totalProfitPct, Double maxDrawdownPct) {
        this.code = code;
        this.timeframe = timeframe;
        this.strategies = strategies;
        this.mainModel = mainModel;
        this.entryThreshold = entryThreshold;
        this.startDate = startDate;
        this.endDate = endDate;
        this.totalTrades = totalTrades;
        this.wins = wins;
        this.draws = draws;
        this.losses = losses;
        this.winRate = winRate;
        this.totalProfitPct = totalProfitPct;
        this.maxDrawdownPct = maxDrawdownPct;
        this.createdAt = LocalDateTime.now();
    }
}


