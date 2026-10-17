---
name: windows-python-stdin
description: >-
  Run multiline Python from Windows PowerShell without broken quoting or Cyrillic source encoding.
---

# Python из Windows PowerShell

В Windows PowerShell 5 передача сложного `python -c '..."..."...'` через native argv может потерять внутренние кавычки. Для многострочного Python используй single-quoted here-string через stdin либо отдельный UTF-8 `.py`. Это не относится к обычному `python -m module`.

`PYTHONUTF8=1` управляет Python, но не кодировкой pipe PowerShell. Для кириллицы в исходнике сначала задай `$OutputEncoding`; иначе используй ASCII escape (`'\u0425'`, `chr(0x425)`). Не повторяй сломанную команду с теми же кавычками.

```powershell
$env:PYTHONUTF8 = '1'
$OutputEncoding = New-Object System.Text.UTF8Encoding($false)
@'
import json
value = "Асылбек"
assert value == "\u0410\u0441\u044b\u043b\u0431\u0435\u043a"
print(json.dumps({"quote": "a\"b", "unicode_ok": True}))
'@ | & '.\.venv\Scripts\python.exe' -
```

В этом repo scripts/*.sh запускаются через Git Bash; это не повод переносить туда операции удаления Windows-путей. Не подставляй непроверенные строки в shell и не печатай окружение/секреты при диагностике. Здесь достаточно проверить exit code и синтетический JSON; банковские строки для проверки кодировки не нужны.
