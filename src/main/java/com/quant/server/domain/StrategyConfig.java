package com.quant.server.domain;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.springframework.cglib.core.Local;

import java.time.LocalDateTime;

@Entity
@Table(name = "strategy_config")
@Getter
@NoArgsConstructor
public class StrategyConfig {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, length = 50)
    private String name;

    @Column(nullable = false)
    private String activeStrategies;

    @Column(length = 20)
    private String mainModel;

    @Column(name = "entry_threshold")
    private Integer entryThreshold;

    @Column(columnDefinition = "TEXT")
    private String parameters;

    @Column(name = "is_active", nullable = false)
    private Boolean isActive;

    @Column(name = "created_at", nullable = false)
    private LocalDateTime createdAt;

    @Column(name = "updated_at")
    private LocalDateTime updatedAt;

    public StrategyConfig(String name, Integer entryThreshold, String mainModel, String activeStrategies, String parameters) {
        this.name = name;
        this.entryThreshold = entryThreshold;
        this.mainModel = mainModel;
        this.activeStrategies = activeStrategies;
        this.parameters = parameters;
        this.isActive = false;
        this.createdAt = LocalDateTime.now();
    }

    public void activate() {
        this.isActive = true;
        this.updatedAt = LocalDateTime.now();
    }

    public void deactivate() {
        this.isActive = false;
        this.updatedAt = LocalDateTime.now();
    }

    public void updateConfig(String activeStrategies, String mainModel,
                             Integer entryThreshold, String parameters) {
        this.activeStrategies = activeStrategies;
        this.mainModel = mainModel;
        this.entryThreshold = entryThreshold;
        this.parameters = parameters;
        this.updatedAt = LocalDateTime.now();
    }
}
