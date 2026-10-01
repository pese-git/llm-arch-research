// Даты для <lastmod> в карте сайта: последний коммит, менявший исходник страницы.
// Дата сборки не подходит — она одинакова у всех страниц и меняется при каждой выкатке.
import { execFileSync } from 'node:child_process';

/**
 * Путь файла от корня репозитория → дата последнего коммита (ISO 8601) для файлов в paths.
 * Один проход `git log`: коммиты идут от новых к старым, первая встреча файла — его дата.
 * Без git (или без истории) — пустой Map: карта сайта собирается без lastmod.
 */
export function lastModified(repoRoot, paths) {
  let log;
  try {
    log = execFileSync('git', ['log', '--format=%x00%cI', '--name-only', '--', ...paths], {
      cwd: repoRoot,
      encoding: 'utf8',
      maxBuffer: 64 * 1024 * 1024,
    });
  } catch (err) {
    console.warn(`[lastmod] git log недоступен (${err.message.split('\n')[0]}) — карта сайта без lastmod`);
    return new Map();
  }
  const dates = new Map();
  for (const commit of log.split('\0').slice(1)) {
    const [date, ...files] = commit.split('\n').filter(Boolean);
    for (const file of files) if (!dates.has(file)) dates.set(file, date);
  }
  return dates;
}
