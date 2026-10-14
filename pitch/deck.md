---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section { font-family: -apple-system, "Segoe UI", Inter, sans-serif; padding: 56px 72px; color: #1d1d1b; }
  h1 { font-size: 46px; letter-spacing: -0.01em; color: #1d1d1b; }
  h2 { font-size: 34px; color: #1d1d1b; }
  strong { color: #2f5fc4; }
  .big { font-size: 96px; font-weight: 700; color: #2f5fc4; line-height: 1; }
  .muted { color: #6b6b66; font-size: 22px; }
  table { font-size: 22px; }
  img { max-height: 430px; }
---

<!--
Шаблон питча (Marp). Заголовок слайда = вывод, одна мысль на слайд, цифры только из runs/*/metrics.json и docs/INSIGHTS.md.
Превью: VS Code + расширение «Marp for VS Code». Экспорт:
  npx -y @marp-team/marp-cli@latest pitch/deck.md --pdf --allow-local-files -o pitch/deck.pdf
  npx -y @marp-team/marp-cli@latest pitch/deck.md --pptx --allow-local-files -o pitch/deck.pptx
Речь и тайминг — docs/PITCH.md.
-->

# <!-- Название решения -->
### <!-- Одна фраза: что получает ForteBank -->

<span class="muted">Команда <!-- название --> · DATATHON'26 · NU Data Science Club × ForteBank</span>

---

## <!-- Проблема: «Банк теряет X на …» -->

<div class="big"><!-- цифра --></div>

<!-- Кто страдает, сколько стоит, как решают сейчас. Источник цифры — мелко внизу. -->

---

## Данные: <!-- что было и что пришлось чинить -->

- <!-- таблицы, строки, период -->
- <!-- что нашли в качестве данных: пропуски, дубли, сдвиг -->
- Валидация: <!-- «учились на прошлом, проверяли на будущем» / «клиент не попадает в обе части» -->

---

## <!-- Инсайт 1: утверждение с цифрой -->

![](../outputs/figures/insight_1.png)

<span class="muted"><!-- действие для банка одной строкой --></span>

---

## <!-- Инсайт 2 -->

![](../outputs/figures/insight_2.png)

---

## <!-- Инсайт 3 -->

![](../outputs/figures/insight_3.png)

---

## Модель: <!-- метрика --> **<!-- значение -->** против <!-- бейзлайн -->

| | <!-- метрика --> |
|---|---:|
| константа (dummy) | |
| бейзлайн | |
| **финал** (<!-- модель/ансамбль -->) | |

<span class="muted">± по фолдам <!-- … -->, run <!-- run_id --></span>

---

## Почему модель так решает

![](../runs/RUN_ID/importance.png)

<!-- 3 фактора человеческими словами: банк обязан объяснять решения клиенту и регулятору -->

---

## Эффект для банка (оценка)

<div class="big"><!-- ≈ N ₸ / −X% --></div>

<!-- формула и допущения мелко: «если …, то …» -->

---

## Как внедрить

- <!-- API: POST /api/predict, страница с прогонами -->
- <!-- где в процессе банка, кто пользователь -->
- <!-- что нужно для пилота -->

---

## Что дальше

- <!-- 2–3 шага развития -->

<span class="muted"><!-- Имена · роли · GitHub-репозиторий --></span>
