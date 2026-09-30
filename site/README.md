# Сайт учебного пособия

Сайт собирается из [`docs/`](../docs) на [Astro](https://astro.build) 7 и [Starlight](https://starlight.astro.build). Источник текста — только `docs/*.md`: там же его читают на GitHub, в `site/` лежат только настройки сборки.

```bash
cd site
npm install
npm run dev       # http://localhost:4321/llm-arch-research/ — правки в docs/ подхватываются сразу
npm run build     # статический сайт в site/dist
npm run preview   # просмотр собранного сайта
```

## Как docs/ превращается в сайт

- [`scripts/sync-docs.mjs`](scripts/sync-docs.mjs) при каждом запуске копирует `docs/*.md` в `src/content/docs/` (в git не хранится):
  - заголовок `# …` становится `title` страницы;
  - строка навигации под ним («← назад · Оглавление · вперёд») убирается — её заменяют боковое меню и ссылки «Назад/Далее»;
  - `README.md` становится главной страницей.
- Боковое меню строится по разделу «Оглавление» в `docs/README.md`: жирная строка (`**Часть I. Основы**`) — группа, первая ссылка на `.md` в строке — пункт. Глава, которой нет в оглавлении, попадает в группу «Прочее».
- [`src/plugins/remark-github-docs.mjs`](src/plugins/remark-github-docs.mjs):
  - формулы в синтаксисе GitHub (`` $`…`$ `` и блоки ` ```math `) рендерит KaTeX;
  - ссылки `глава.md#якорь` ведут на страницы сайта;
  - ссылки на код (`../llm/src/...`) ведут на файлы в ветке `master` на GitHub.
- Диаграммы ` ```mermaid ` рисует в браузере [astro-mermaid](https://github.com/joesaby/astro-mermaid), со светлой и тёмной темой.
- Поиск — Pagefind, встроен в Starlight.

Меню читается при старте: после правки оглавления в `docs/README.md` перезапустите `npm run dev`.

## Публикация

Workflow [`.github/workflows/docs-site.yml`](../.github/workflows/docs-site.yml):
- в PR, где меняются `docs/` или `site/`, проверяет сборку;
- после слияния в `master` публикует сайт на GitHub Pages — `https://pese-git.github.io/llm-arch-research/`.

Pages нужно один раз включить в настройках репозитория: Settings → Pages → Source: GitHub Actions.

Адрес для другого хостинга задают переменные окружения `SITE_URL` и `SITE_BASE`, например `SITE_BASE=/ npm run build` для корня домена.
