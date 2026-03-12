package com.quant.server.repository;

import com.quant.server.domain.StrategyConfig;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface StrategyConfigRepository extends JpaRepository<StrategyConfig, Long>{

    Optional<StrategyConfig> findByIsActiveTrue();

    List<StrategyConfig> findAllByOrderByCreatedAtDesc();

    Optional<StrategyConfig> findByName(String name);
}
