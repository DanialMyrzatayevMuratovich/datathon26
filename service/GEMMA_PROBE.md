# Изолированный Gemma challenger, 18.10.2026

Асылбек запросил другую готовую visual-модель и разрешил работу над общим качеством. Используем **Google Gemma 4 E2B Instruct QAT Q4_0**, не обучение с нуля и не перенос адаптера Qwen. Подтверждённого дообученного командой Qwen-адаптера в production нет. Детекторы, правила, MLX backend и зависимости проекта пока не заменены.

Официальный источник: [Google GGUF](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf), revision `675cff42a74c774d6cb76f76d8eacb49b48c9b93`. Два файла суммарно 4 336 349 920 байт, SHA256 проверены:

| Файл | SHA256 |
|---|---|
| `gemma-4-E2B_q4_0-it.gguf` | `fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634` |
| `gemma-4-E2B-it-mmproj.gguf` | `021059cce659fe7f9170d5599761d7bbaf644b798dab9503aca30dc43e6beb14` |

Runtime — официальный [llama.cpp b11715](https://github.com/ggml-org/llama.cpp/releases/tag/b11715), Windows CUDA12.4, build commit `9d1878a43`. ZIP и CUDA runtime проверены по опубликованным release SHA256. Модель, projector и runtime находятся вне Git: `../models/gemma-4-e2b-it-qat-q4_0`, `../local_ml/llama-b11715`.

## Что измерено

`runs/jobs/gemma-e2b-20261018`: 20 GPU layers, CPU projector, max560 image tokens. Три синтетических изображения 128×128 правильно определены как Red/Green/Blue за 9.390с. Первый T9 action-probe X-1/X-3, every8, без кэша: оба запланированных набора окон завершились, 6 VLM-запросов за 71.546с, весь цикл 249.953с. Три ответа не соответствовали строгому формату A/B/C/D. Это техническая проба, **не** измерение бинарной accuracy.

`runs/jobs/gemma-e2b-gpu-20261018`: те же 20 GPU layers, projector на CUDA, min=max560 image tokens. Три синтетических изображения 640×480 правильно определены за 4.266с; наблюдаемая VRAM2579MiB, температура59°C. Входное разрешение и token budget изменены, поэтому эти два smoke не являются строгим сравнением скорости при одинаковом входе.

Первый balanced8 остановился по общему дедлайну1200с: сохранено6/8 видео,77 запросов. В Windows System зарегистрированы Kernel-Power506 в03:09:25 и507 в03:17:36 (Modern Standby). Медиана серверных запросов с учётом одного synthetic smoke —2.713с, но один занял272.422с и совпал с периодом сна. Такой прогон не является чистым измерением производительности. A встречался и в положительном, и в чистом видео; правило «A → нарушение» не принимается.

Начат свежий uncached balanced8 в `runs/jobs/gemma-e2b-awake-20261018`. На время эксперимента wrapper держит [SetThreadExecutionState](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate) SYSTEM+DISPLAY_REQUIRED, возвращая предыдущее состояние в finally. Настройки схемы питания не меняются; ручной сон/закрытие крышки этот API не блокирует. Есть синтетический тест восстановления после исключения.

Для второго прогона задана grammar A/B/C/D: это ограничение синтаксиса; D сохраняет возможность неопределённого ответа. В runtime-отчёт добавлены SHA256 фактических JPEG-входов и вопросов. После исправления wrapper также сохраняет их после каждого успешного запроса в `.inputs.jsonl`, чтобы timeout не уничтожал provenance завершённых вызовов. Это позволяет проверять совпадение входов при дальнейшем сравнении; изображения и свободные ответы не экспортируются.

Подготовлена поддержка одного большего кандидата — [официального E4B QAT GGUF](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf), revision `4b4a2c1d584be7264f87aac328a1bc739ce81b6c`,6 146 493 536байт. Его SHA256 закреплены в `PREPARED_MODELS`, выбор `--model gemma-4-e4b-it-qat-q4_0`. Загрузка ещё проверяется; выполнение и преимущество E4B не подтверждены. У E4B собственный projector, подставлять E2B-файл нельзя. Запускать только после остановки E2B с отдельной проверкой RAM/VRAM.

## Воспроизведение

На подготовленном Windows-хосте запустить в отдельном PowerShell из корня repo:

```powershell
& '..\local_ml\llama-b11715\llama-server.exe' -m '..\models\gemma-4-e2b-it-qat-q4_0\gemma-4-E2B_q4_0-it.gguf' --mmproj '..\models\gemma-4-e2b-it-qat-q4_0\gemma-4-E2B-it-mmproj.gguf' --host 127.0.0.1 --port 8091 --alias gemma-4-e2b-it-qat-q4_0 -c 4096 -np 1 -ngl 20 --mmproj-offload --image-min-tokens 560 --image-max-tokens 560 --cache-ram 0 --no-cache-idle-slots --offline --no-context-shift -t 2 -tb 2 --reasoning off --reasoning-budget 0
```

После `/health` = ok, в другом терминале:

```powershell
uv run python -m service.local_vlm_probe --only X-1,X-3,X-4,X-5,X-13,X-19,X-44,X-37 --videos data/raw/open/open/videos --out runs/jobs/gemma-e2b-gpu-20261018/t9-balanced.json --every 4 --max-calls 40 --budget-sec 1200
```

Нужны существующие зависимости проекта, FFmpeg/ffprobe и локальный `../models/grounding-dino-tiny` закреплённой проектом ревизии. Сначала закрыть другие тяжёлые jobs. После прогона остановить **свой** llama-server через Ctrl+C. Дедлайн wrapper завершает Python/FFmpeg и очищает собственный scratch, но не управляет сервером, запущенным в другом терминале. После timeout/failure также остановить сервер перед следующим экспериментом.

Эксперимент обращается только к127.0.0.1, игнорирует HTTP proxies, запрещает redirects; загрузка моделей offline, банковские кадры остаются локально. Родитель проверяет SHA256 подготовленных весов, но identity работающего сервера нужно подтверждать отдельно. В выполненном запуске рядом сохранены PID, executable, точные аргументы и хеши executable/DLL (`server-launch.json`).

## Условия принятия

T9 action-probe различает ручной счёт, использование машинки, перемещение пачки и неопределённость. **Ручной счёт не равен автоматически нарушению; использование машинки не доказывает пересчёт всей пачки.** Он сам по себе не закрывает частичный пересчёт и не оценивает отсутствующие production-ветки T3/T6/T13. `complete` означает завершение назначенных окон, не непрерывное наблюдение операции.

Переход в production возможен после реализации нужного правила и двух полных uncached40 с одинаковой разметкой, контроля всех19 чистых видео, совместного совпадения типа/времени и реального API/SLA. Для сравнения именно моделей должны совпадать JPEG-хеши, вопросы, окна и настройки генерации; иначе сравниваются два pipeline. 95%=38/40 development-вердиктов, ещё не подтверждено и не гарантирует результат на скрытых видео.
