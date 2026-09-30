# Сайт документации

Сайт собирается из [`docs/`](../docs) на [Astro](https://astro.build) 7 и [Starlight](https://starlight.astro.build). Источник текста — только `docs/**/*.md`: там же его читают на GitHub, в `site/` лежат только настройки сборки.

```bash
cd site
npm install
npm run dev       # http://localhost:4321/llm-arch-research/ — правки в docs/ подхватываются сразу
npm run build     # статический сайт в site/dist
npm run preview   # просмотр собранного сайта
```

## Как docs/ превращается в сайт

- [`scripts/sync-docs.mjs`](scripts/sync-docs.mjs) при каждом запуске копирует `docs/**/*.md` в `src/content/docs/` с той же структурой папок (в git не хранится):
  - заголовок `# …` становится `title` страницы;
  - строка навигации под ним («← назад · Оглавление · вперёд») убирается — её заменяют боковое меню и ссылки «Назад/Далее»;
  - `docs/<раздел>/README.md` становится первой страницей раздела;
  - `docs/README.md` на сайт не попадает: это входная страница для GitHub, на сайте её место занимает визитка.
- Боковое меню — три группы: «Учебное пособие», «Руководство пользователя», «Для разработчиков» (список — `sections` в `sync-docs.mjs`). Каждая строится по разделу «Оглавление» в `docs/<раздел>/README.md`: жирная строка (`**Часть I. Основы**`) — подгруппа, первая ссылка на `.md` в строке — пункт. Глава, которой нет в оглавлении, попадает в подгруппу «Прочее».
- [`src/plugins/remark-github-docs.mjs`](src/plugins/remark-github-docs.mjs):
  - формулы в синтаксисе GitHub (`` $`…`$ `` и блоки ` ```math `) рендерит KaTeX;
  - ссылки разрешаются относительно исходного файла: `глава.md#якорь` или `../guide/training.md` ведут на страницы сайта;
  - ссылки на код (`../../llm/src/...`) ведут на файлы в ветке `master` на GitHub.
- Диаграммы ` ```mermaid ` рисует в браузере [astro-mermaid](https://github.com/joesaby/astro-mermaid), со светлой и тёмной темой.
- Поиск — Pagefind, встроен в Starlight.

Меню читается при старте: после правки оглавления в `docs/<раздел>/README.md` перезапустите `npm run dev`.

## Главная страница

Главная — визитка проекта [`src/landing/index.mdx`](src/landing/index.mdx): splash-шаблон Starlight без бокового меню, со встроенными компонентами (`Card`, `LinkCard`, `Tabs`). Она пишется только для сайта, поэтому лежит в `site/`, а не в `docs/`; `sync-docs.mjs` копирует её в коллекцию как `index.mdx`. Ссылки в ней — относительные адреса страниц сайта (`textbook/gpt/`), а не пути к `.md`: плагин ссылок `.mdx` не трогает.

Логотип — [`src/assets/logo.svg`](src/assets/logo.svg) (шапка и визитка) и его копия `public/favicon.svg` (иконка вкладки).

## Публикация

Workflow [`.github/workflows/docs-site.yml`](../.github/workflows/docs-site.yml):
- в PR, где меняются `docs/` или `site/`, проверяет сборку;
- после слияния в `master` публикует сайт на GitHub Pages — `https://pese-git.github.io/llm-arch-research/`.

Pages нужно один раз включить в настройках репозитория: Settings → Pages → Source: GitHub Actions.

Адрес для другого хостинга задают переменные окружения `SITE_URL` и `SITE_BASE`, например `SITE_BASE=/ npm run build` для корня домена.
