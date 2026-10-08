# Сайт документации

Сайт собирается из [`docs/`](../docs) на [Astro](https://astro.build) 7 и [Starlight](https://starlight.astro.build). Источник текста — только `docs/**/*.md`: там же его читают на GitHub, в `site/` лежат только настройки сборки.

```bash
cd site
npm install
npm run dev       # http://localhost:4321/ — правки в docs/ подхватываются сразу
npm run build     # статический сайт в site/dist
npm run preview   # просмотр собранного сайта
```

## Как docs/ превращается в сайт

- [`scripts/sync-docs.mjs`](scripts/sync-docs.mjs) при каждом запуске копирует `docs/**/*.md` в `src/content/docs/` с той же структурой папок, а `docs/assets/` (иллюстрации из `docs/tools/figures.py`) — в `src/content/docs/assets/`, чтобы относительные ссылки на картинки в главах работали и Astro отдавал их как ресурсы (в git не хранится):
  - заголовок `# …` становится `title` страницы;
  - комментарий `<!-- description: … -->` под заголовком становится `description` — описанием страницы для поисковиков и превью ссылок (`og:description`); на GitHub он не виден. Без него `sync-docs` предупреждает, и страница получает общее описание сайта;
  - строка навигации под ним — одни ссылки через «·»: «← назад · Оглавление · вперёд» в главах, соседние разделы в README — убирается, её заменяют боковое меню и ссылки «Назад/Далее»;
  - ссылка «Редактировать страницу» (`editUrl`) ведёт на исходник в `docs/` на GitHub, а не на сгенерированный файл; адрес репозитория и ветка — `repoUrl` и `branch` в `sync-docs.mjs`;
  - `docs/<раздел>/README.md` становится первой страницей раздела; оглавление справа у неё — только если в ней от трёх разделов `##`;
  - `docs/README.md` на сайт не попадает: это входная страница для GitHub, на сайте её место занимает визитка.
- Боковое меню — три группы: «Учебное пособие», «Руководство пользователя», «Для разработчиков» (список — `sections` в `sync-docs.mjs`). Первый пункт группы — README раздела под подписью из `overview` («О пособии» и т. п.: в «Назад/Далее» название группы не видно). Остальные строятся по разделу «Оглавление» в `docs/<раздел>/README.md`: жирная строка (`**Часть I. Основы**`) — подгруппа, первая ссылка на `.md` в строке — пункт. Глава, которой нет в оглавлении, попадает в подгруппу «Прочее».
- [`src/plugins/remark-github-docs.mjs`](src/plugins/remark-github-docs.mjs):
  - формулы в синтаксисе GitHub (`` $`…`$ `` и блоки ` ```math `) рендерит KaTeX;
  - ссылки разрешаются относительно исходного файла: `глава.md#якорь` или `../guide/training.md` ведут на страницы сайта;
  - ссылки на код (`../../llm/src/...`) ведут на файлы в ветке `master` на GitHub;
  - подпись ссылки, которая совпадает с именем файла (`[attention.md](attention.md)`), заменяется названием страницы — её заголовком `# …`.
- Диаграммы ` ```mermaid ` рисует в браузере [astro-mermaid](https://github.com/joesaby/astro-mermaid), со светлой и тёмной темой.
- Карта сайта — `@astrojs/sitemap`, подключена явно ради `<lastmod>`: дата последнего коммита исходника страницы (`docs/…` или визитки), её считает [`scripts/lastmod.mjs`](scripts/lastmod.mjs) по `git log`. Без git сайт собирается без `<lastmod>`.
- Поиск — Pagefind, встроен в Starlight. Вес страницы в результатах задаёт `searchWeights` в `sync-docs.mjs`: у бэклога он понижен до 0.2, иначе журнал, где упомянут почти каждый термин, выходит выше глав. Проверять поиск нужно на собранном сайте (`npm run build && npm run preview`): в `npm run dev` индекса нет.
- `robots.txt` собирает [`src/pages/robots.txt.ts`](src/pages/robots.txt.ts): в нём абсолютный адрес `sitemap-index.xml`, зависящий от `SITE_URL` и `SITE_BASE`.

Меню читается при старте: после правки оглавления в `docs/<раздел>/README.md` перезапустите `npm run dev`.

Astro кэширует отрисованные страницы в `node_modules/.astro` и сбрасывает кэш, только когда меняется текст страницы. После правки плагинов в `src/plugins/` или `astro.config.mjs` удалите кэш: `rm -rf node_modules/.astro`. В CI сборка идёт с чистого листа, и кэш ей не мешает.

## Главная страница

Главная — визитка проекта [`src/landing/index.mdx`](src/landing/index.mdx): splash-шаблон Starlight без бокового меню, со встроенными компонентами (`Card`, `LinkCard`, `Tabs`). Она пишется только для сайта, поэтому лежит в `site/`, а не в `docs/`; `sync-docs.mjs` копирует её в коллекцию как `index.mdx`. Ссылки в ней — относительные адреса страниц сайта (`textbook/gpt/`), а не пути к `.md`: плагин ссылок `.mdx` не трогает.

Логотип — [`src/assets/logo.svg`](src/assets/logo.svg) (шапка и визитка) и его копия `public/favicon.svg` (иконка вкладки).

Картинка превью ссылок (`og:image`, 1200×630) — [`src/assets/og.svg`](src/assets/og.svg). PNG для соцсетей и мессенджеров собирается из него командой `npm run og-image` в `public/og.png` и хранится в git: шрифты берутся из системы, и в CI текст отрисовался бы иначе. После правки `og.svg` пересоберите PNG. Адрес картинки в `astro.config.mjs` абсолютный и учитывает `SITE_URL` и `SITE_BASE`.

## Публикация

Сайт публикуется Docker-образом на кластере — `https://llm-arch-research.openidealab.com` (раздел «Docker» ниже). Workflow [`.github/workflows/docs-site.yml`](../.github/workflows/docs-site.yml) проверяет сборку в PR, где меняются `docs/` или `site/`, а после слияния в `master` ещё и собирает образ и публикует его в Harbor. Выкатка на кластер — руками.

По умолчанию сайт собирается для корня этого домена. Адрес для другого хостинга задают переменные окружения `SITE_URL` и `SITE_BASE`, например `SITE_URL=https://example.org SITE_BASE=/docs npm run build`.

## Docker

[`Dockerfile`](Dockerfile) собирает сайт в образ nginx. Собирать из **корня репозитория**: сайту нужны `docs/` и файлы репозитория для ссылок на код.

```bash
docker buildx build --platform linux/amd64 -f site/Dockerfile \
  -t harbor.openidealab.com/llm-arch-research/site:latest .
docker run --rm -p 8080:80 harbor.openidealab.com/llm-arch-research/site:latest   # http://localhost:8080/
```

- Сборка в два этапа: Node собирает сайт, nginx ([`nginx.conf`](nginx.conf)) раздаёт `dist/` на порту 80. Образ — около 30 МБ.
- По умолчанию сайт собирается для корня домена `https://llm-arch-research.openidealab.com` (`SITE_BASE=/`). Другой адрес — `--build-arg SITE_URL=… --build-arg SITE_BASE=…`.
- Что попадает в контекст сборки, задаёт [`Dockerfile.dockerignore`](Dockerfile.dockerignore): без виртуальных окружений, `checkpoints/` и результатов сборки. `.git` нужен: по нему считаются даты `<lastmod>`, в образ nginx он не попадает.
- Образ публикуется в Harbor: `harbor.openidealab.com/llm-arch-research/site`, теги — короткий SHA коммита и `latest`. Это делает job `publish` в [`docs-site.yml`](../.github/workflows/docs-site.yml) на каждый push в `master`, меняющий `docs/` или `site/` (или по кнопке Run workflow на `master`). Ему нужны секреты репозитория `HARBOR_USERNAME` и `HARBOR_PASSWORD` — робот-аккаунт Harbor с правом push в проект `llm-arch-research`; без них job только пишет предупреждение. CI собирает с полной историей git, поэтому даты `<lastmod>` в карте сайта верные.
- Собрать и отправить образ руками (например, из worktree без полной истории — тогда карта сайта будет без `<lastmod>`): `docker login harbor.openidealab.com` с ролью Developer или выше в проекте и `docker push` тегов SHA и `latest`.
- Выкатка на кластер не автоматизирована: Deployment `site` в namespace `llm-arch-research` держит образ по тегу SHA, и после публикации тег нужно обновить — `kubectl -n llm-arch-research set image deployment/site site=harbor.openidealab.com/llm-arch-research/site:<sha>`. Откат — `kubectl rollout undo`.
