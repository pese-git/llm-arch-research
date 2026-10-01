// Переносит docs/**/*.md в коллекцию Starlight (src/content/docs) и строит боковое меню
// по оглавлениям разделов. Исходники остаются в docs/ — они же читаются на GitHub;
// сгенерированные файлы в git не хранятся.
//
// Структура docs/ сохраняется: docs/<раздел>/README.md — первая страница раздела,
// docs/<раздел>/<глава>.md — страница <раздел>/<глава>/. Главная — визитка src/landing/index.mdx
// (только для сайта); docs/README.md — входная страница для GitHub, на сайт не попадает.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const siteDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const docsDir = path.resolve(siteDir, '../docs');
export const outDir = path.join(siteDir, 'src/content/docs');
// Репозиторий и ветка, на которые ведут ссылки на код и «Редактировать страницу»
export const repoUrl = 'https://github.com/pese-git/llm-arch-research';
export const branch = 'master';
const sidebarFile = path.join(siteDir, 'src/generated/sidebar.json');
const landingFile = path.join(siteDir, 'src/landing/index.mdx');

// Разделы в порядке меню: папка в docs/, подпись группы
// и подпись первой страницы раздела — в «Назад/Далее» она видна без названия группы
export const sections = [
  { dir: 'textbook', label: 'Учебное пособие', overview: 'О пособии' },
  { dir: 'guide', label: 'Руководство пользователя', overview: 'О руководстве' },
  { dir: 'dev', label: 'Для разработчиков', overview: 'О разделе для разработчиков' },
];

/** Путь .md относительно docs/ (через «/») → slug страницы: README.md — индекс папки. */
export const slugOf = (rel) => rel.replace(/(^|\/)README\.md$/, '$1').replace(/\.md$/, '').replace(/\/$/, '');

/** Путь .md относительно docs/ → файл в коллекции: README.md → index.md. */
const outName = (rel) => rel.replace(/(^|\/)README\.md$/, '$1index.md');

const yamlString = (s) => JSON.stringify(s);

/** Все .md в docs/ — пути относительно docs/ через «/»; docs/README.md заменяет визитка. */
export function listDocs() {
  const out = [];
  const walk = (dir) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) walk(full);
      else if (entry.name.endsWith('.md')) out.push(path.relative(docsDir, full).split(path.sep).join('/'));
    }
  };
  walk(docsDir);
  return out.filter((f) => f !== 'README.md').sort();
}

/** Строка навигации — только ссылки через «·»: «← назад · Оглавление · вперёд» в главах, ссылки на соседние разделы в README. */
const isNavLine = (line) => /^\[[^\]]+\]\([^)]+\)(\s*·\s*\[[^\]]+\]\([^)]+\))+\s*$/.test(line.trim());

/**
 * Вес страниц в поиске Pagefind (по умолчанию 1): бэклог упоминает почти каждый термин
 * и без понижения выходит в результатах выше глав, где термин разобран.
 */
const searchWeights = { 'dev/backlog.md': 0.2 };

/** Описание страницы — HTML-комментарий под заголовком: на GitHub он не виден. */
const descriptionRe = /^<!--\s*description:\s*([\s\S]+?)\s*-->\s*$/;

/**
 * Markdown главы → страница Starlight: заголовок первого уровня становится `title`,
 * комментарий `<!-- description: … -->` — `description` (поисковики и превью ссылок),
 * `editUrl` ведёт на исходник в docs/ на GitHub, а не на сгенерированный файл,
 * строка навигации под ним убирается — её заменяют боковое меню и ссылки «Назад/Далее»
 * внизу страницы. Оглавление справа у README раздела — только если в нём от трёх разделов.
 */
function toPage(rel, source) {
  const lines = source.split('\n');
  const h1 = lines.findIndex((l) => l.startsWith('# '));
  if (h1 === -1) throw new Error(`${rel}: нет заголовка первого уровня`);
  const title = lines[h1].slice(2).trim();
  let body = lines.slice(h1 + 1);
  let description = null;
  const descLine = body.findIndex((l) => descriptionRe.test(l));
  if (descLine !== -1 && body.slice(0, descLine).every((l) => !l.startsWith('## '))) {
    description = body[descLine].match(descriptionRe)[1].replace(/\s+/g, ' ');
    body.splice(descLine, 1);
  } else {
    console.warn(`[sync-docs] docs/${rel}: нет <!-- description: … --> под заголовком — у страницы будет общее описание сайта`);
  }
  const firstText = body.findIndex((l) => l.trim() !== '');
  if (firstText !== -1 && (body[firstText].includes('[Оглавление](README.md)') || isNavLine(body[firstText]))) {
    body = body.slice(firstText + 1);
  }
  const front = ['---', `title: ${yamlString(title)}`];
  if (description) front.push(`description: ${yamlString(description)}`);
  front.push(`editUrl: ${yamlString(`${repoUrl}/edit/${branch}/docs/${rel}`)}`);
  const sectionCount = body.filter((l) => l.startsWith('## ')).length;
  if (rel.endsWith('README.md') && sectionCount < 3) front.push('tableOfContents: false');
  front.push('---', '');
  let text = body.join('\n').replace(/^\n+/, '');
  // Pagefind применяет вес ко всему тексту внутри элемента; пустые строки вокруг — чтобы внутри разбирался markdown
  if (searchWeights[rel]) text = `<div data-pagefind-weight="${searchWeights[rel]}">\n\n${text}\n\n</div>\n`;
  return front.join('\n') + text;
}

/**
 * Меню раздела — по его README, раздел «Оглавление»: жирная строка `**Часть I. Основы**`
 * открывает подгруппу, первая ссылка на .md в каждой строке списка или таблицы — пункт;
 * номер главы из первой колонки таблицы добавляется к названию. Пункты до первой жирной
 * строки идут без подгруппы. Главы, которых нет в оглавлении, — в подгруппу «Прочее».
 */
function buildSection({ dir, label, overview }, files) {
  const readmePath = path.join(docsDir, dir, 'README.md');
  if (!fs.existsSync(readmePath)) return null;
  const readme = fs.readFileSync(readmePath, 'utf8');
  const start = readme.indexOf('## Оглавление');
  if (start === -1) throw new Error(`docs/${dir}/README.md: нет раздела «Оглавление»`);
  const end = readme.indexOf('\n## ', start + 1);
  const items = [{ label: overview, slug: dir }];
  let group = null;
  for (const line of readme.slice(start, end === -1 ? undefined : end).split('\n')) {
    const heading = line.match(/^\*\*(.+)\*\*\s*$/);
    if (heading) {
      group = { label: heading[1], items: [] };
      items.push(group);
      continue;
    }
    const link = line.match(/\[([^\]]+)\]\(([\w.-]+\.md)\)/);
    if (!link || link[2] === 'README.md') continue;
    const num = line.match(/^\|\s*(\d+)\s*\|/);
    const item = { label: num ? `${num[1]}. ${link[1]}` : link[1], slug: slugOf(`${dir}/${link[2]}`) };
    (group ? group.items : items).push(item);
  }
  // Подгруппа без пунктов (например, ссылки только за пределы раздела) в меню не нужна
  for (let i = items.length - 1; i >= 0; i--) if (items[i].items && !items[i].items.length) items.splice(i, 1);
  const listed = new Set();
  const collect = (list) => list.forEach((i) => (i.items ? collect(i.items) : listed.add(i.slug)));
  collect(items);
  const missing = files
    .filter((f) => f.startsWith(`${dir}/`))
    .map(slugOf)
    .filter((s) => !listed.has(s));
  if (missing.length) items.push({ label: 'Прочее', items: missing.map((slug) => ({ slug })) });
  return { label, items };
}

export function syncDocs({ quiet = false } = {}) {
  const files = listDocs();
  fs.rmSync(outDir, { recursive: true, force: true });
  for (const rel of files) syncFile(rel);
  fs.copyFileSync(landingFile, path.join(outDir, 'index.mdx'));
  const sidebar = [{ label: 'Главная', link: '/' }, ...sections.map((s) => buildSection(s, files)).filter(Boolean)];
  fs.mkdirSync(path.dirname(sidebarFile), { recursive: true });
  fs.writeFileSync(sidebarFile, JSON.stringify(sidebar, null, 2) + '\n');
  if (!quiet) console.log(`[sync-docs] ${files.length} файлов из docs/`);
}

/** rel — путь .md относительно docs/ через «/». */
export function syncFile(rel) {
  const source = fs.readFileSync(path.join(docsDir, rel), 'utf8');
  const out = path.join(outDir, outName(rel));
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, toPage(rel, source));
}

export function removeFile(rel) {
  fs.rmSync(path.join(outDir, outName(rel)), { force: true });
}

if (process.argv[1] === fileURLToPath(import.meta.url)) syncDocs();
