# data/raw — файлы организаторов

Кладём сюда то, что выдали на открытии (CSV, Parquet, Excel), **в git не попадает** (конфиденциальные данные банка; проверяет `.gitignore` и `scripts/secret-scan.sh`). Напарнику — флешкой, AirDrop или через общий Drive, не через репозиторий. Для Colab — копия в `MyDrive/datathon26/raw/`.

Пути и target прописываются в `task.toml`; без них оркестратор берёт первые `train*` / `test*` / `sample_submission*` отсюда.
