// @ts-check
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import sitemap from '@astrojs/sitemap';
import starlight from '@astrojs/starlight';
import mermaid from 'astro-mermaid';
import rehypeKatex from 'rehype-katex';
import remarkGithubDocs from './src/plugins/remark-github-docs.mjs';
import { branch, docsDir, listDocs, outDir, removeFile, repoUrl, slugOf, syncDocs, syncFile } from './scripts/sync-docs.mjs';
import { lastModified } from './scripts/lastmod.mjs';

const siteDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(siteDir, '..');

// Сайт живёт в корне домена на кластере (Docker-образ, site/Dockerfile).
// Для другого хостинга — SITE_URL и SITE_BASE (например, SITE_BASE=/docs для подпути).
const site = process.env.SITE_URL ?? 'https://llm-arch-research.openidealab.com';
const base = process.env.SITE_BASE ?? '/';
// Картинка превью ссылок — public/og.png (npm run og-image); соцсети требуют абсолютный адрес
const ogImage = new URL(`${base.replace(/\/$/, '')}/og.png`, site).href;

syncDocs();

// <lastmod> в карте сайта: адрес страницы → её исходник → дата последнего коммита.
// Главная — визитка из site/, остальные страницы — из docs/
const pageSources = new Map(listDocs().map((rel) => [slugOf(rel), `docs/${rel}`]));
pageSources.set('', 'site/src/landing/index.mdx');
const lastmod = lastModified(repoRoot, [...pageSources.values()]);
const basePath = base.endsWith('/') ? base : `${base}/`;
const sitemapWithLastmod = sitemap({
  serialize(item) {
    const slug = new URL(item.url).pathname.slice(basePath.length).replace(/\/$/, '');
    const date = lastmod.get(pageSources.get(slug));
    if (date) item.lastmod = date;
    return item;
  },
});
const sidebar = JSON.parse(fs.readFileSync(path.join(siteDir, 'src/generated/sidebar.json'), 'utf8'));

/** Путь файла → путь относительно docs/ через «/», или null, если файл не .md из docs/. */
const docsRel = (/** @type {string} */ file) => {
  const rel = path.relative(docsDir, file);
  return rel.startsWith('..') || !file.endsWith('.md') ? null : rel.split(path.sep).join('/');
};

/** В режиме dev пересобирает страницу при изменении файла в docs/. */
const docsWatcher = {
  name: 'docs-watcher',
  hooks: {
    'astro:server:setup': ({ server }) => {
      server.watcher.add(docsDir);
      const onChange = (/** @type {string} */ file) => {
        const rel = docsRel(file);
        if (!rel || rel === 'README.md') return;
        // Оглавления README задают меню — меню читается при старте, нужен перезапуск
        if (rel.endsWith('README.md')) syncDocs({ quiet: true });
        else syncFile(rel);
      };
      server.watcher.on('add', onChange);
      server.watcher.on('change', onChange);
      server.watcher.on('unlink', (file) => {
        const rel = docsRel(file);
        if (rel) removeFile(rel);
      });
    },
  },
};

export default defineConfig({
  site,
  base,
  trailingSlash: 'always',
  integrations: [
    docsWatcher,
    // Своя карта сайта вместо встроенной в Starlight (та же, но с lastmod): Starlight
    // не добавляет свою, если @astrojs/sitemap уже подключён
    sitemapWithLastmod,
    // До Starlight: плагин должен забрать блоки ```mermaid раньше подсветки кода
    mermaid({ theme: 'default', autoTheme: true }),
    starlight({
      title: 'LLM Arch Research',
      logo: { src: './src/assets/logo.svg' },
      favicon: '/favicon.svg',
      description:
        'GPT-1, GPT-2, LLaMA, Mistral, Mixtral и Gemma на PyTorch: учебное пособие, руководство пользователя и документация для разработчиков.',
      defaultLocale: 'root',
      locales: { root: { label: 'Русский', lang: 'ru' } },
      social: [{ icon: 'github', label: 'GitHub', href: repoUrl }],
      head: [
        { tag: 'meta', attrs: { property: 'og:image', content: ogImage } },
        { tag: 'meta', attrs: { property: 'og:image:width', content: '1200' } },
        { tag: 'meta', attrs: { property: 'og:image:height', content: '630' } },
        { tag: 'meta', attrs: { property: 'og:image:alt', content: 'LLM Arch Research — архитектуры LLM с нуля на PyTorch' } },
        { tag: 'meta', attrs: { name: 'twitter:card', content: 'summary_large_image' } },
        { tag: 'meta', attrs: { name: 'twitter:image', content: ogImage } },
      ],
      sidebar,
      customCss: ['katex/dist/katex.min.css', './src/styles/custom.css'],
      tableOfContents: { minHeadingLevel: 2, maxHeadingLevel: 3 },
      lastUpdated: false,
      pagination: true,
    }),
  ],
  markdown: {
    // unified (remark/rehype), а не Sätteri по умолчанию: нужны свои плагины для формул и ссылок
    processor: unified({
      remarkPlugins: [[remarkGithubDocs, { base, docsDir, contentDir: outDir, repoRoot, repoUrl, branch }]],
      rehypePlugins: [[rehypeKatex, { strict: 'ignore', throwOnError: false }]],
    }),
  },
});
