package com.quant.server.domain;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Entity
@Table(name = "stock_price",
       uniqueConstraints = @UniqueConstraint(
               columnNames = {"code", "timeframe", "trade_time"}))
@Getter
@NoArgsConstructor
public class StockPrice {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long Id;

    @Column(nullable = false, length = 20)
    private String code;

    @Column(nullable = false)
    private Double open;

    @Column(nullable = false)
    private Double high;

    @Column(nullable = false)
    private Double low;

    @Column(name = "close_price", nullable = false)
    private Double close;

    @Column(nullable = false)
    private Long volume;

    @Column(nullable = false, length = 10)
    private String timeframe;

    @Column(name = "trade_time", nullable = false)
    private LocalDateTime tradeTime;

    public StockPrice(String code, Double open, Double high, Double low, Double close, Long volume, String timeframe, LocalDateTime tradeTime) {
        this.tradeTime = tradeTime;
        this.timeframe = timeframe;
        this.close = close;
        this.high = high;
        this.open = open;
        this.code = code;
        this.low = low;
        this.volume = volume;
    }
}
