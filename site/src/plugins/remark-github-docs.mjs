// Markdown из docs/ написан под GitHub. Плагин приводит его к сайту:
//
// 1. Формулы GitHub: строчные $`…`$ и блоки ```math — в узлы, которые рендерит rehype-katex
//    (классы math-inline / math-display). Обычный remark-math не подходит: он не понимает
//    обратные кавычки и принял бы за формулу любой одиночный «$» в тексте.
// 2. Ссылки: `attention.md#якорь` → страница сайта, `README.md` → главная,
//    `../llm/src/...` и прочие пути репозитория → файл или папка на GitHub.
import fs from 'node:fs';
import path from 'node:path';
import { visit, SKIP } from 'unist-util-visit';

const mathNode = (value, display) => ({
  type: display ? 'mathDisplay' : 'mathInline',
  value,
  data: {
    hName: display ? 'div' : 'span',
    hProperties: { className: ['math', display ? 'math-display' : 'math-inline'] },
    hChildren: [{ type: 'text', value }],
  },
});

function convertMath(tree) {
  visit(tree, 'code', (node, index, parent) => {
    if (node.lang !== 'math' || !parent) return;
    parent.children[index] = mathNode(node.value, true);
    return SKIP;
  });
  // $`x`$ разбирается как текст «…$», inlineCode «x», текст «$…»
  visit(tree, (node) => {
    if (!Array.isArray(node.children)) return;
    const children = node.children;
    for (let i = 1; i < children.length - 1; i++) {
      const [prev, cur, next] = [children[i - 1], children[i], children[i + 1]];
      if (
        cur.type === 'inlineCode' &&
        prev.type === 'text' && prev.value.endsWith('$') &&
        next.type === 'text' && next.value.startsWith('$')
      ) {
        prev.value = prev.value.slice(0, -1);
        next.value = next.value.slice(1);
        children[i] = mathNode(cur.value, false);
      }
    }
  });
}

export default function remarkGithubDocs({ base = '/', docsDir, repoRoot, repoUrl, branch = 'master' }) {
  const siteBase = base.endsWith('/') ? base : `${base}/`;
  const docFiles = new Set(fs.readdirSync(docsDir).filter((f) => f.endsWith('.md')));

  function rewrite(url) {
    if (!url || /^([a-z][a-z0-9+.-]*:|#|\/)/i.test(url)) return url; // внешние, якоря, абсолютные
    const [target, hash = ''] = url.split(/(?=#)/);
    const anchor = hash ? decodeURIComponent(hash) : '';
    const file = decodeURIComponent(target);
    if (docFiles.has(file)) {
      const slug = file === 'README.md' ? '' : `${file.replace(/\.md$/, '')}/`;
      return `${siteBase}${slug}${anchor}`;
    }
    // Путь внутри репозитория: docs/<url> → относительно корня
    const rel = path.posix.normalize(path.posix.join('docs', file)).replace(/\/$/, '');
    if (rel.startsWith('..')) return url;
    let kind = 'blob';
    try {
      if (fs.statSync(path.join(repoRoot, rel)).isDirectory()) kind = 'tree';
    } catch {
      // файла нет в рабочей копии — всё равно ведём на GitHub
    }
    return `${repoUrl}/${kind}/${branch}/${rel.split('/').map(encodeURIComponent).join('/')}${hash}`;
  }

  return (tree) => {
    convertMath(tree);
    visit(tree, ['link', 'definition'], (node) => {
      node.url = rewrite(node.url);
    });
  };
}
