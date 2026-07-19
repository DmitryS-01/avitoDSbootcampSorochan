# Ноутбуки

Ноутбуки запускаются по порядку из корня репозитория в ядре `Python (.venv)`:

1. `00_eda_cleaning.ipynb` — данные и очистка HTML;
2. `01_popularity_baseline.ipynb` — тривиальный baseline;
3. `02_bm25.ipynb` — лексический поиск;
4. `03_query_knn.ipynb` — перенос меток похожих запросов;
5. `04_linear_svc.ipynb` — линейный текстовый классификатор;
6. `05_lambdamart.ipynb` — learning-to-rank;
7. `06_semantic_query_knn.ipynb` — multilingual E5;
8. `07_ranker_blend.ipynb` — E5/LightGBM и noise/spell blend;
9. `08_error_analysis.ipynb` — анализ ошибок основной ветки;
10. `09_final_pipeline.ipynb` — проверка и сохранение submission.

В `00–08` клетка сохранения закомментирована. В `09` она активна.
