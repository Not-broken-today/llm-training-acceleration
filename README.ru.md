> 🇷🇺 Русский | 🇬🇧 [English](README.md)

# 🚀 Ускорение обучения LLM: Сравнение оптимизаторов

> **Решение технического задания «LLM Training Acceleration»**  
> Сравнительный анализ оптимизаторов AdamW, Muon и гибридного подхода для LoRA-дообучения модели Qwen2.5-0.5B

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📋 Обзор проекта

Данный проект реализует сравнительное исследование стратегий оптимизации при параметрически-эффективном дообучении больших языковых моделей. Цель — оценить **скорость обучения**, **эффективность использования памяти GPU** и **качество финальной модели** для трёх оптимизаторов:

| Оптимизатор | Описание |
|-------------|----------|
| **AdamW** | Стандартный адаптивный оптимизатор |
| **Muon** | Оптимизатор на основе ортогонализации матриц (локальная реализация) |
| **Hybrid** | Гибридный подход: 50% 2D-параметров → Muon, остальные → AdamW |

### 🔬 Параметры эксперимента

- **Модель**: [`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- **Датасет**: [`Elriggs/openwebtext-100k`](https://huggingface.co/datasets/Elriggs/openwebtext-100k) (1% = 1 000 сэмплов)
- **Оценка**: [PIQA](https://github.com/EleutherAI/lm-evaluation-harness) (логический вывод)
- **Fine-tuning**: LoRA (`r=16`, `alpha=32`, `dropout=0.05`)
- **Точность**: `bfloat16` (нативная поддержка на RTX 40/50-серии)
- **Фреймворк**: Hugging Face `transformers` + `trl` (SFTTrainer)

> ⚠️ **Примечание**: Оптимизатор MeZO **не реализован** в данной версии (задание повышенной сложности отложено).

---

## ✨ Особенности

- ✅ **Гибридный оптимизатор**: автоматическое разделение параметров (50% 2D-тензоров → Muon, остальные → AdamW)
- ✅ **Фильтрация параметров**: исключение `embed_tokens` и `lm_head` из группы Muon
- ✅ **Эффективное использование памяти**: gradient checkpointing + bf16 + LoRA
- ✅ **Детальное логирование**: потери, LR, память, время на каждом шаге
- ✅ **Автоматическая оценка на PIQA** через `lm-evaluation-harness`
- ✅ **Кросс-платформенные скрипты настройки** (`setup.sh` / `setup.bat`)

---

## 📦 Установка

### Требования

- Python **3.12** (требуется скриптами настройки)
- Видеокарта с CUDA и ≥8 ГБ VRAM (рекомендуется 16 ГБ для `batch_size=4`)
- Git

### Быстрый старт

```bash
# Клонирование репозитория
git clone https://github.com/Not-broken-today/llm-training-acceleration.git  
cd llm-training-acceleration

# Запуск настройки (создаёт venv + устанавливает зависимости)
# Для Linux/macOS:
chmod +x setup.sh && ./setup.sh
# Для Windows:
setup.bat

# Активация окружения (если не активировано автоматически)
# Для Linux/macOS:
source .venv_llm/bin/activate
# Для Windows CMD:
.venv_llm\Scripts\activate
```

### Зависимости

Основные пакеты указаны в `requirements.txt`:
```txt
torch>=2.5.0
transformers>=4.45.0
trl>=0.12.0
datasets>=3.0.0
lm-eval>=0.4.3
peft>=0.13.0
# + утилиты
```

> 💡 Кастомная реализация Muon включена локально — установка из внешнего репозитория не требуется.

---

## ⚙️ Конфигурация

Отредактируйте `config.yaml` для настройки экспериментов:

```yaml
# Активный оптимизатор: "adamw" | "muon" | "hybrid"
active_optimizer: "adamw"

model:
  model_name: "Qwen/Qwen2.5-0.5B"
  torch_dtype: "bfloat16"  # Используйте "float16" для старых GPU

lora:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

training:
  num_train_epochs: 5
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 5e-5
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.03
  bf16: true
  gradient_checkpointing: true

optimizers:
  muon:
    ns_steps: 5
    momentum: 0.95
    nesterov: true
  adamw:
    betas: [0.9, 0.999]
    eps: 1e-8
```

---

## ▶️ Запуск экспериментов

### Один оптимизатор

```bash
# Запуск с AdamW (по умолчанию)
python src/main.py

# Запуск с Muon
python src/main.py active_optimizer=muon

# Запуск с гибридным оптимизатором
python src/main.py active_optimizer=hybrid
```

### Через готовые скрипты

```bash
# Linux/macOS
./run_file/AdamW_run.sh
./run_file/Muon_run.sh
./run_file/Hybrid_run.sh

# Windows
run_file\AdamW_run.bat
run_file\Moun_run.bat
run_file\Hybrid_run.bat
```

### Результаты

Результаты сохраняются в `outputs/checkpoints/`:
```
outputs/
├── checkpoints/
│   ├── metrics_{optimizer}.json   # Метрики на каждом шаге
│   ├── results_{optimizer}.json   # Итоговая сводка + оценка на PIQA
│   └── checkpoint-*/              # Веса LoRA (формат PEFT)
└── YYYY-MM-DD/
        └── hh-mm-ss/
            └── main.log           # Сохраненные логи
```

---

## 📊 Результаты (фактические данные)

*Эксперименты проведены на RTX 5060 Ti 16 ГБ, Windows 11, Python 3.12*

| Оптимизатор | Время обучения | Пиковая память | Финальный loss | Точность на PIQA | Шагов/сек |
|-------------|---------------|----------------|----------------|------------------|-----------|
| **AdamW** | 7м 41с | 3912 МБ | 2,863 | **70,29%** | 0,682 |
| **Muon** | 9м 22с | 3846 МБ | 2,903 | 70,13% | 0,560 |
| **Hybrid** | 8м 23с | 3878 МБ | 2,888 | 69,86% | 0,626 |

### 🔍 Ключевые наблюдения

1. **Память**: Muon использует на ~2–3% меньше VRAM (один буфер момента против двух у AdamW)
2. **Скорость**: AdamW быстрее всех; замедление Muon обусловлено итерациями Ньютона-Шульца
3. **Качество**: Все оптимизаторы показывают сопоставимую точность на PIQA (~70%) при малом объёме данных
4. **Стабильность**: Гибридный подход демонстрирует плавную сходимость, комбинируя преимущества обоих методов

---

## 🗂️ Структура проекта

```
llm-training-acceleration/
├── 📄 README.md                 # Этот файл
├── 📄 config.yaml               # Конфигурация экспериментов
├── 📄 requirements.txt          # Зависимости Python
├── 📄 setup.py / setup.sh / setup.bat  # Скрипты установки
├── 📄 mypy.ini / pylint_test_code.*    # Проверка типов и линтинг
│
├── 📁 src/                      # Исходный код
│   ├── __init__.py
│   ├── main.py                  # Точка входа: пайплайн обучения
│   ├── model.py                 # Загрузка модели + применение LoRA
│   ├── dataset.py               # Загрузка и токенизация датасета
│   │
│   ├── 📁 optimizers/
│   │   ├── __init__.py
│   │   ├── optimizer.py         # Фабрика: выбор/создание оптимизатора
│   │   │   ├── create_optimizer(name, params, **kwargs)
│   │   │   ├── create_trainer_config(**kwargs)
│   │   │   └── create_trainer(...)
│   │   └── muon.py              # Реализация Muon
│   │
│   ├── 📁 utilities/
│   │   ├── loggers/             # MetricsCallback, мониторинг памяти
│   │   └── system/              # Утилиты для GPU, очистка
│   │
│   └── 📁 evaluation/
│       └── piqa_eval.py         # Оценка на датасете PIQA
│
├── 📁 run_file/                 # Готовые скрипты запуска
│   ├── AdamW_run.sh / .bat
│   ├── Muon_run.sh / .bat
│   ├── Hybrid_run.sh / .bat
│   └── All_run.sh / .bat        # Последовательный запуск всех трёх
│
└── 📁 outputs/                  # Генерируется при обучении (в .gitignore)
    ├── checkpoints/
    └── 📁 YYYY-MM-DD/
        └── 📁 hh-mm-ss/
            └── 📄 main.log

```

---

## 🔧 Логика выбора оптимизатора (`src/optimizers/optimizer.py`)

### Фильтрация параметров

```python
# Параметры для Muon: 2D+ тензоры, кроме embed_tokens и lm_head
muon_params = [
    p for p in params
    if p.ndim >= 2 and "embed_tokens" not in str(p) and "lm_head" not in str(p)
]

# Остальные параметры → AdamW
adamw_params = [p for p in params if p not in muon_params]
```

### Гибридный режим

```python
# Все 2D-параметры (исключая embed_tokens/lm_head)
all_2d_params = [...]  # как выше

# Делим пополам
split_idx = len(all_2d_params) // 2
muon_params = all_2d_params[:split_idx]           # первая половина → Muon
adamw_params = all_2d_params[split_idx:] + other_params  # вторая половина + остальное → AdamW
```

---

## 🧪 Качество кода

Проект использует:
- **`mypy`** для статической проверки типов (настройки в `mypy.ini`)
- **`pylint`** для линтинга (вспомогательный скрипт `pylint_test_code.sh / .bat`)

Запуск проверок:
```bash
# Linux/macOS
chmod +x pylint_test_code.sh && ./pylint_test_code.sh

# Windows
pylint_test_code.bat

# или вручную:
pylint src/**/*.py
```

---

## 📝 План работ / Известные ограничения

- [ ] Реализовать оптимизатор MeZO (задание повышенной сложности)

---

## 📜 Лицензия

Распространяется под лицензией **MIT**. Подробности см. в файле [`LICENSE`](LICENSE).

---

## 🙏 Благодарности

- [MoonshotAI / Muon](https://github.com/MoonshotAI/Moonlight) — реализации оптимизатора Muon
- [Princeton NLP / MeZO](https://github.com/princeton-nlp/MeZO) — оптимизация без градиентов
- [Hugging Face](https://huggingface.co) — transformers, datasets, TRL, PEFT
- [EleutherAI](https://github.com/EleutherAI/lm-evaluation-harness) — фреймворк для оценки моделей

---

*Последнее обновление: март 2026 г.*
