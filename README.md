# Avito Help Search

Для каждого запроса из `test.f` я ранжирую статьи справки и
возвращаю 10 `article_id`. Целевая метрика — `MAP@10`.

## Результаты

| Подход | OOF MAP@10 | TestSys |
|---|---:|---:|
| Popularity baseline | 0.3159 | 0.27 |
| Stemmed BM25 | 0.3187 | — |
| Query kNN | 0.6008 | — |
| LinearSVC | 0.6005 | — |
| XGBoost pairwise | 0.6947 | — |
| LightGBM bagged | 0.6974 | — |
| **Ranker RRF ensemble** | **0.6996** | **0.601** |

## Подход

1. Из HTML я извлекаю видимый текст, заголовки, ссылки, `alt` и начало
   статьи. Отдельно строю stemmed-представление.
2. Кандидаты и признаки получаю из BM25, word/char TF-IDF, query kNN,
   линейных моделей, профилей запросов и связей между статьями.
3. Для каждой пары `запрос — статья` считаю score-, rank-, overlap- и
   length-признаки.
4. Финальный порядок я получаю RRF-ансамблем трёх LightGBM LambdaMART
   и XGBoost pairwise.

В RRF я использую веса `0.4644`, `0.0108`, `0.4767` и `0.0481` для
LightGBM seed 42, LightGBM seed 123, XGBoost и LightGBM без row bagging.
Я подобрал их по сохранённым OOF-предсказаниям и оставил комбинацию с
максимальным OOF MAP@10. Я смешиваю ранги, а не raw-score, потому что
шкалы моделей различаются. Дополнительные LightGBM я не убираю: с ними
ансамбль даёт 0.6996 против 0.6988 у пары лучших ranker-ов.

## Валидация

Я использую 5-fold OOF с `shuffle=True` и `random_state=42`. Query-признаки
для каждого fold строятся только по его train-части. `test.f` не
участвует в обучении и подборе. Веса RRF подобраны по OOF, поэтому
0.6996 не является оценкой на отдельном holdout.

Я использую только выданные Feather-файлы: без внешних API,
предобученных моделей, данных и ручной разметки `test.f`.

## Запуск

Все команды и ноутбуки я запускаю из корня репозитория. Окружение
я проверил на Python `3.14.6`; версии библиотек зафиксированы.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

```bash
python baseline.py  # popularity baseline
python run.py       # финальное решение
```

В обоих скриптах я проверяю формат и сохраняю `outputs/answer.csv`.
В ноутбуках `00–08` клетка сохранения закомментирована. В `09_final_pipeline.ipynb`
я оставляю её активной: именно этот pipeline создаёт submission с TestSys 0.601.
Я возвращаю ровно 10 статей: это допустимый максимум,
который не снижает MAP@10 при уже зафиксированном порядке верхних документов.

## Структура

```text
baseline.py          popularity baseline
run.py               запуск финального решения
src/                 очистка, retrieval, признаки и ranker-ы
notebooks/           отобранные эксперименты
outputs/answer.csv   готовый submission
statement/           условие и данные
```
