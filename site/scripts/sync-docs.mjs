// Переносит docs/*.md в коллекцию Starlight (src/content/docs) и строит боковое меню
// по оглавлению docs/README.md. Исходники остаются в docs/ — они же читаются на GitHub;
// сгенерированные файлы в git не хранятся.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const siteDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const docsDir = path.resolve(siteDir, '../docs');
const outDir = path.join(siteDir, 'src/content/docs');
const sidebarFile = path.join(siteDir, 'src/generated/sidebar.json');

// README.md → главная страница, остальные файлы — по имени без .md
export const slugOf = (file) => (file === 'README.md' ? '' : file.replace(/\.md$/, ''));

const yamlString = (s) => JSON.stringify(s);

/**
 * Markdown главы → страница Starlight: заголовок первого уровня становится `title`,
 * строка навигации под ним («← назад · Оглавление · вперёд») убирается — её заменяют
 * боковое меню и ссылки «Назад/Далее» внизу страницы.
 */
function toPage(file, source) {
  const lines = source.split('\n');
  const h1 = lines.findIndex((l) => l.startsWith('# '));
  if (h1 === -1) throw new Error(`${file}: нет заголовка первого уровня`);
  const title = lines[h1].slice(2).trim();
  let body = lines.slice(h1 + 1);
  const firstText = body.findIndex((l) => l.trim() !== '');
  if (firstText !== -1 && body[firstText].includes('[Оглавление](README.md)')) {
    body = body.slice(firstText + 1);
  }
  const front = ['---', `title: ${yamlString(title)}`];
  if (file === 'README.md') front.push('tableOfContents: false');
  front.push('---', '');
  return front.join('\n') + body.join('\n').replace(/^\n+/, '');
}

/**
 * Группы меню — по оглавлению README: жирная строка `**Часть I. Основы**` открывает группу,
 * первая ссылка на .md в каждой строке списка или таблицы — пункт; номер главы из первой
 * колонки таблицы добавляется к названию.
 */
function buildSidebar(readme, files) {
  const start = readme.indexOf('## Оглавление');
  const end = readme.indexOf('\n## ', start + 1);
  if (start === -1) throw new Error('docs/README.md: нет раздела «Оглавление»');
  const groups = [];
  for (const line of readme.slice(start, end === -1 ? undefined : end).split('\n')) {
    const heading = line.match(/^\*\*(.+)\*\*\s*$/);
    if (heading) {
      groups.push({ label: heading[1], items: [] });
      continue;
    }
    const link = line.match(/\[([^\]]+)\]\(([\w.-]+\.md)\)/);
    if (!link || !groups.length) continue;
    const num = line.match(/^\|\s*(\d+)\s*\|/);
    if (link[2] === 'README.md') continue;
    groups.at(-1).items.push({
      label: num ? `${num[1]}. ${link[1]}` : link[1],
      slug: slugOf(link[2]),
    });
  }
  const listed = new Set(groups.flatMap((g) => g.items.map((i) => i.slug)));
  const missing = files.map(slugOf).filter((s) => s && !listed.has(s));
  if (missing.length) groups.push({ label: 'Прочее', items: missing.map((slug) => ({ slug })) });
  return [{ label: 'О пособии', link: '/' }, ...groups];
}

export function syncDocs({ quiet = false } = {}) {
  const files = fs.readdirSync(docsDir).filter((f) => f.endsWith('.md')).sort();
  fs.rmSync(outDir, { recursive: true, force: true });
  fs.mkdirSync(outDir, { recursive: true });
  for (const file of files) syncFile(file);
  const readme = fs.readFileSync(path.join(docsDir, 'README.md'), 'utf8');
  fs.mkdirSync(path.dirname(sidebarFile), { recursive: true });
  fs.writeFileSync(sidebarFile, JSON.stringify(buildSidebar(readme, files), null, 2) + '\n');
  if (!quiet) console.log(`[sync-docs] ${files.length} файлов из docs/`);
}

export function syncFile(file) {
  const source = fs.readFileSync(path.join(docsDir, file), 'utf8');
  const name = file === 'README.md' ? 'index.md' : file;
  fs.writeFileSync(path.join(outDir, name), toPage(file, source));
}

export function removeFile(file) {
  fs.rmSync(path.join(outDir, file === 'README.md' ? 'index.md' : file), { force: true });
}

if (process.argv[1] === fileURLToPath(import.meta.url)) syncDocs();
