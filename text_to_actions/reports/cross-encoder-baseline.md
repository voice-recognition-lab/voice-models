**Cross-encoder baseline — 21 November 2026**

- Модель `cointegrated/rubert-tiny2`, без дообучения, со случайно инициализированной скалярной scoring-головой. Шагов оптимизатора — 0; seed 42. Это исходный BERT с новой головой, а не обученный reranker.
- Ревизия `e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae`.
- Данные `data/v1.1`, весь `validation`: 1 653 примера, 55 кандидатов на команду. Режим `all`, без отбора кандидатов bi-encoder.
- Существующее разбиение без кластеризации перефразировок. Команды русские, описания действий английские. Модель оценивает пару «команда + схема действия» и ранжирует действия по logit.

Результаты:

- Accuracy / Recall@1 — **0,36%**.
- Recall@3 — **3,33%**.
- Recall@5 — **5,32%**.
- Recall@10 — **17,54%**.
- Recall@20 — **40,17%**.
- Recall@30 — **53,90%**.

Оценка всей выборки заняла 512,18 с (8 мин 32 с), в среднем 309,85 мс на пример.
Время включает оценку пар и запись предсказаний; загрузка модели и подготовка каталога исключены.
Условия запуска — CPU, 4 потока, batch size 16 пар, max length 512, seed 42.
Python 3.12.12, PyTorch 2.15.0, Transformers 4.57.6.

Команда запуска из корня репозитория:

```bash
source .venv/bin/activate
python -m text_to_actions.cross_encoder.evaluate.run --config text_to_actions/cross_encoder/configs/encoder.json --mode all --output text_to_actions/runs/cross_encoder_baseline
```

[Метрики](../runs/cross_encoder_baseline/20261121T170217007782Z/metrics.json),
[конфигурация запуска](../runs/cross_encoder_baseline/20261121T170217007782Z/config.json),
[предсказания](../runs/cross_encoder_baseline/20261121T170217007782Z/validation.predictions.jsonl).
