# Личный бот вакансий (wet-lab + postdoc)

Надстройка над PhD Sky для форка: бот ищет посты о вакансиях в Bluesky,
отсеивает «не вакансии» с помощью LLM, оставляет биологию (включая мокрую:
молекулярная и клеточная биология) и присылает дайджест в Telegram.
Supabase, сайт и почтовые рассылки для этого не нужны.

Как устроен дайджест:

1. **⭐ In focus** — эпигенетика/хроматин, 3D-геном (Hi-C), нейроны. Полная
   карточка с текстом поста. Внутри раздела сначала postdoc-уровень.
2. **🔬 Postdoc & research staff** — postdoc, research fellow / associate,
   staff scientist, research assistant по остальной биологии. Короткая запись.
3. **🏢 Industry R&D** — исследовательские позиции в компаниях (scientist,
   research associate в биотехе и фарме). Короткая запись, тег `#Industry`.
4. **🎓 PhD & other** — PhD и магистерские позиции. Короткая запись.

Индустриальная вакансия по вашим темам попадает в первый раздел, тоже с тегом
`#Industry`.

Повторы одной вакансии (репосты, напоминания, та же ссылка на подачу)
отбрасываются.

## Настройка (один раз, ~15 минут)

### 1. Ключи

| Секрет | Где взять |
| --- | --- |
| `BLUESKY_HANDLE` | Хэндл аккаунта Bluesky, например `masha.bsky.social` |
| `BLUESKY_PASSWORD` | **App password**, не основной пароль: Bluesky → Settings → Privacy and security → App passwords |
| `MISTRAL_API_KEY` | Ключ API в консоли Mistral (console.mistral.ai → API Keys). Вместо него подойдёт `GROQ_API_KEY` (console.groq.com/keys), `GEMINI_API_KEY` или `NVIDIA_API_KEY` — достаточно любого одного. С Mistral бот работает заметно быстрее: у Groq и Gemini между запросами пауза 12–15 секунд, поэтому первый проход за 14 дней растянется на несколько запусков |
| `TELEGRAM_BOT_TOKEN` | В Telegram: @BotFather → `/newbot` → токен |
| `TELEGRAM_CHANNEL_ID` | Куда слать, см. ниже |

`TELEGRAM_CHANNEL_ID` — два варианта:

- **Личный чат с ботом.** Напишите своему боту `/start`, затем откройте в
  браузере `https://api.telegram.org/bot<ТОКЕН>/getUpdates` и возьмите число из
  `"chat":{"id": ...}`.
- **Канал.** Создайте канал, добавьте бота администратором и укажите
  `@имя_канала`.

### 2. Секреты в GitHub

В форке: **Settings → Secrets and variables → Actions → New repository secret** —
добавьте пять секретов из таблицы (вкладка *Secrets*, не *Variables*).

### 3. Запуск

1. Вкладка **Actions** → разрешить workflows в форке.
2. Слева **My Job Bot** → **Enable workflow** (в форках расписания выключены
   по умолчанию).
3. **Run workflow** с галочкой `dry_run` — дайджест появится в логе, в Telegram
   ничего не уйдёт. Удобно, чтобы проверить ключи и посмотреть на выдачу.
4. **Run workflow** без галочки — первый настоящий запуск. Он просматривает
   последние 14 дней, поэтому идёт дольше и пришлёт больше обычного.

Дальше бот запускается сам три раза в день (08:23, 14:23, 20:23 по Москве) и
присылает только новое. Состояние хранится в `mybot_data/state.json`, архив
отправленных вакансий — в `mybot_data/positions.csv` (GitHub показывает его
таблицей с поиском). Оба файла бот коммитит сам.

## Как подстроить под себя

Всё настраивается в одном файле — [`src/personal/profile.py`](src/personal/profile.py):

- `EXTRA_QUERIES` — поисковые запросы Bluesky. Добавьте строку, чтобы искать
  новую тему.
- `INCLUDE_INDUSTRY` — `False`, если позиции из индустрии не нужны;
  `INDUSTRY_QUERIES` — поисковые запросы для них.
- `FOCUS_TOPICS` — темы со звёздочкой (регулярные выражения).
- `WET_LAB_TOPICS` — теги MolBio / CellBio.
- `RELEVANT_DISCIPLINES` — сейчас `{"Biology"}`: проходит вся биология. Если
  выдача слишком большая, поставьте `set()` — останутся только вакансии,
  совпавшие с темами из двух списков выше.
- `PRIORITY_POSITION_TYPES` — что считать «научными сотрудниками».
- `EXCLUDED_POSITION_TYPES` — например `{"Master Student"}`, чтобы убрать
  магистерские позиции.
- `COMPACT_NON_FOCUS` — `False`, если нужны полные карточки для всех вакансий.

Расписание — в `.github/workflows/my-job-bot.yml` (`cron`, время в UTC).

Локальная проверка:

```bash
pip install -e .
python -m src.personal.run --dry-run   # ключи берутся из .env
python -m pytest tests/test_personal_bot.py
```

## Что изменено относительно upstream

- `src/personal/` — сам бот; `tests/test_personal_bot.py` — тесты.
- `.github/workflows/my-job-bot.yml` — расписание бота (работает только в форках).
- В четырёх upstream-workflow добавлено условие `if: github.repository == ...`:
  им нужен Supabase оригинального проекта, в форке они просто пропускаются.
- Код и промпт upstream не тронуты. Дополнительные правила для модели
  (research fellow / research associate / staff scientist — уровень Postdoc,
  молекулярная и клеточная нейробиология — Biology, позиции в индустрии и поле
  `sector`) добавляются к промпту на лету в `src/personal/llm.py`.

Обновления из оригинального репозитория подтягиваются кнопкой **Sync fork**.
