package com.quant.server.service;

import com.quant.server.domain.StrategyConfig;
import com.quant.server.repository.StrategyConfigRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Map;

@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class StrategyConfigService {

    private final StrategyConfigRepository strategyConfigRepository;

    @Transactional
    public StrategyConfig create(Map<String, Object> params) {
        StrategyConfig config = new StrategyConfig(
                (String) params.get("name"),
                (Integer) params.get("entryThreshold"),
                (String) params.get("parameters"),
                (String) params.get("activeStrategies"),
                (String) params.get("mainModel")
        );
        return strategyConfigRepository.save(config);
    }

    public List<StrategyConfig> getAll() {
        return strategyConfigRepository.findAllByOrderByCreatedAtDesc();
    }

    public StrategyConfig getById(Long id) {
        return strategyConfigRepository.findById(id)
                .orElseThrow(() -> new RuntimeException(
                        "설정을 찾을 수 없습니다. id=" + id));
    }

    public StrategyConfig getActive() {
        return strategyConfigRepository.findByIsActiveTrue()
                .orElseThrow(() -> new RuntimeException(
                        "활성화된 설정이 없습니다."));
    }

    @Transactional
    public StrategyConfig update(Long id, Map<String,Object> params) {
        StrategyConfig config = strategyConfigRepository.findById(id).orElseThrow(() -> new RuntimeException("설정을 찾을 수 없습니다. id=" + id));
        config.updateConfig(
                (String) params.get("activeStrategies"),
                (String) params.get("mainModel"),
                (Integer) params.get("entryThreshold"),
                (String) params.get("parameters"));
        return config;
    }

    @Transactional
    public StrategyConfig activate(Long id) {
        strategyConfigRepository.findByIsActiveTrue().ifPresent(StrategyConfig::deactivate);
        StrategyConfig config = strategyConfigRepository.findById(id).orElseThrow(() -> new RuntimeException("설정을 찾을 수 없습니다. id=" + id));
        config.activate();
        return config;
    }

    @Transactional
    public void delete(Long id) {
        strategyConfigRepository.deleteById(id);
    }

}
