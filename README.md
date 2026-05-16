# Marketplace Product Matching

Решение задачи бинарной классификации матчинга предложений продавцов с товарами маркетплейса. Для каждой пары `offer` - `goods` модель определяет, является ли пара совпадением (`1`) или нет (`0`).

В решении используются табличные признаки, текстовые и image-эмбеддинги. Для эмбеддингов применяется PCA-снижение размерности, после чего рассчитываются метрики сходства между offer и goods. Также строятся дополнительные признаки и используется pseudo-labeling.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.2-150458?logo=pandas)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2-00A4FF)
![Scikit--learn](https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikit-learn)
![NumPy](https://img.shields.io/badge/NumPy-2.1-013243?logo=numpy)

## Описание задачи

В соревновании необходимо решить финальный этап пайплайна матчинга:

- на вход подаётся пара объектов: товар продавца (`offer`) и товар маркетплейса (`goods`);
- для каждой пары уже заранее найдены кандидаты на матч;
- требуется классифицировать, является ли пара совпадением или нет;
- качество оценивается по `F-score`.

Данные включают:

- табличные признаки: цена, категория, длина текстов, score от рескоринговой модели;
- текстовые эмбеддинги;
- image-эмбеддинги.

## Ноутбуки

### `01_eda.ipynb`

Exploratory Data Analysis:

- структура train/test;
- пропуски и дубли;
- баланс целевого класса;
- распределения числовых признаков;
- анализ цен;
- анализ категорий;
- повторяемость `offer`/`goods`;
- train/test shift;
- покрытие эмбеддингов.

### `02_modeling.ipynb`

Основной обучающий пайплайн:

1. загрузка train/test;
2. загрузка и маппинг текстовых и image-эмбеддингов;
3. PCA-снижение размерности эмбеддингов;
4. расчёт distance-features между `offer` и `goods`;
5. построение price-features и дополнительных инженерных признаков;
6. обучение `CatBoostClassifier`;
7. pseudo-labeling уверенных объектов из test;
8. подбор threshold по validation;
9. сохранение модели и формирование submission.

## Подход к решению

Решение построено как табличный ML pipeline с мультимодальными признаками.

### Работа с эмбеддингами

Для каждого объекта используются:

- текстовые эмбеддинги;
- image-эмбеддинги.

Для снижения размерности:

- текстовые векторы сжимаются через `PCA` до 32 компонент;
- image-векторы сжимаются через `PCA` до 64 компонент.

### Метрики расстояния и сходства

Для пары `offer` и `goods` рассчитываются признаки сходства:

- `manhattan`;
- `euclidean`;
- `cosine`;
- `correlation`;
- `canberra`;
- `pearson`;
- `spearman`;
- `chebyshev`.

Метрики считаются отдельно для текста и изображений. Дополнительно используются взаимодействия:

- произведения `title` и `image` distance-features;
- произведения `attrs+title_score` и distance-features.

### Дополнительные признаки

- `price_diff`;
- `price_ratio`;
- `log_offer_price`;
- `log_goods_price`;
- `log_price_diff`;
- `price_diff_log_ratio`;
- `cat_mean_target`;
- `has_both_images`;
- `is_much_cheaper`;
- `is_much_more_expensive`;
- `price_ratio_rank_in_offer`;
- `price_ratio_log`.

## Модель

В качестве классификатора используется `CatBoostClassifier`.

Основные параметры:

- `iterations=1000`;
- `eval_metric='F1'`;
- `early_stopping_rounds=100`;
- `random_state=42`.

Для оценки качества train разбивается на train/validation через `train_test_split(..., stratify=y)`. Метрика на validation - `F1-score`.

## Pseudo-labeling

После базового обучения модель предсказывает вероятности для test. Затем выбираются уверенные объекты:

- `proba > 0.99` - positive;
- `proba < 0.01` - negative.

Эти объекты добавляются в обучающую выборку с весом `0.5`, после чего модель дообучается на расширенном train.

## Результат

Финальная модель:

- использует мультимодальные эмбеддинги;
- сочетает PCA-признаки, distance-features и price-features;
- обучается на CatBoost;
- использует pseudo-labeling и подбор порога.

Итоговый Score на Kaggle: **0.92651648** (5 место).

## Структура проекта

```text
marketplace-product-matching/
├── data/
├── models/
├── results/
├── 01_eda.ipynb
├── 02_modeling.ipynb
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

### Локальная структура данных

```text
data/
├── train.csv
├── test.csv
├── offer_title_vectors/
│   └── offer_title_vectors/
│       ├── items_deperson.npy
│       └── embed_deperson.npy
├── offer_image_vectors/
│   └── offer_image_vectors/
│       ├── items_deperson.npy
│       └── embed_deperson.npy
├── goods_title_vectors/
│   └── goods_title_vectors/
│       ├── items_deperson.npy
│       └── embed_deperson.npy
└── goods_image_vectors/
    └── goods_image_vectors/
        ├── items_deperson.npy
        └── embed_deperson.npy
```

## Идеи улучшения

- Ансамбль нескольких моделей.
- Больше признаков взаимодействия между текстом, изображениями и ценами.
- Более агрессивный pseudo-labeling или self-training loop.
