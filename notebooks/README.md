# Notebooks

| Ноутбук | Содержание |
|---|---|
| `00_eda_cleaning.ipynb` | EDA и очистка HTML |
| `01_popularity_baseline.ipynb` | popularity baseline |
| `02_bm25.ipynb` | BM25 и stemming |
| `03_query_knn.ipynb` | query kNN |
| `04_linear_svc.ipynb` | LinearSVC |
| `05_lambdamart.ipynb` | LightGBM LambdaMART |
| `06_xgboost_ranker.ipynb` | XGBoost pairwise |
| `07_ranker_blend.ipynb` | RRF-ансамбль |
| `08_error_analysis.ipynb` | анализ OOF-ошибок |
| `09_final_pipeline.ipynb` | проверка и сохранение submission |

Все ноутбуки я запускаю из корня через kernel `Python (.venv)`.
В `00–08` клетку сохранения `answer.csv` я оставляю закомментированной. В `09`
клетка активна и создаёт submission, загруженный в TestSys.
