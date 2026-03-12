package com.quant.server.repository;

import com.quant.server.domain.BacktestResult;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface BacktestResultRepository extends JpaRepository<BacktestResult, Long>{
    List<BacktestResult> findByCodeOrderByCreatedAtDesc(String code);

    List<BacktestResult> findByCodeOrderByTotalProfitPctDesc(String code);

    List<BacktestResult> findByMainModelOrderByTotalProfitPctDesc(String mainModel);
}
