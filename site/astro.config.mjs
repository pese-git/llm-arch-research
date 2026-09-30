// @ts-check
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import starlight from '@astrojs/starlight';
import mermaid from 'astro-mermaid';
import rehypeKatex from 'rehype-katex';
import remarkGithubDocs from './src/plugins/remark-github-docs.mjs';
import { docsDir, outDir, removeFile, syncDocs, syncFile } from './scripts/sync-docs.mjs';

const siteDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(siteDir, '..');
const repoUrl = 'https://github.com/pese-git/llm-arch-research';

// GitHub Pages проекта: https://pese-git.github.io/llm-arch-research/.
// Для другого хостинга — SITE_URL и SITE_BASE (например, SITE_BASE=/ для корня домена).
const site = process.env.SITE_URL ?? 'https://pese-git.github.io';
const base = process.env.SITE_BASE ?? '/llm-arch-research';

syncDocs();
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
      remarkPlugins: [[remarkGithubDocs, { base, docsDir, contentDir: outDir, repoRoot, repoUrl, branch: 'master' }]],
      rehypePlugins: [[rehypeKatex, { strict: 'ignore', throwOnError: false }]],
    }),
  },
});
