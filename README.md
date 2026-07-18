# Avito Help Search: popularity baseline

Baseline для поиска статей справочного центра. Для каждого запроса решение
возвращает 10 `article_id`; целевая метрика — `MAP@10`.

## Подход

Статьи ранжируются по частоте появления в `ground_truth` из `calibration.f`.
При равной частоте выше ставится меньший `article_id`, поэтому результат
детерминирован. Один top-10 используется для всех тестовых запросов.

Тексты статей и запросов на этом этапе не обрабатываются: baseline нужен как
простая нижняя граница для следующих моделей. Внешние данные и разметка
`test.f` не используются.

## Валидация

| Подход | MAP@10 |
|---|---:|
| Popularity baseline | 0.3158 |

Используется 5-fold OOF с `shuffle=True` и `random_state=42`. Для каждого фолда
частоты считаются только по обучающей части, поэтому валидационные метки не
попадают в признаки.

## Запуск

Все команды и ноутбук запускаются из корня репозитория. Решение проверено на
Python `3.14.6`; версии библиотек зафиксированы в `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python baseline.py
```

Скрипт повторно считает OOF-метрику, проверяет формат submission и сохраняет
`outputs/answer.csv`. Тот же эксперимент доступен в
`notebooks/01_popularity_baseline.ipynb`; клетка сохранения submission по
умолчанию закомментирована.

## Submission

Готовый `outputs/answer.csv` содержит строки для всех 500 `query_id` из
`test.f` в исходном порядке. В каждой строке ровно 10 уникальных существующих
`article_id`. CSV содержит только требуемые колонки `query_id` и `answer`.

## Структура

```text
.
├── baseline.py             # валидация и генерация submission
├── notebooks/              # ноутбук с baseline
├── outputs/answer.csv      # готовый submission
├── statement/              # условие и исходные данные
└── requirements.txt        # зафиксированные зависимости
```
