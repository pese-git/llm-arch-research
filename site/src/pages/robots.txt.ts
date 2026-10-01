// robots.txt с адресом карты сайта: адрес абсолютный и зависит от SITE_URL и SITE_BASE,
// поэтому файл собирается при сборке, а не лежит в public/.
import type { APIRoute } from 'astro';

export const GET: APIRoute = ({ site }) => {
  const sitemap = new URL('sitemap-index.xml', new URL(import.meta.env.BASE_URL, site));
  return new Response(`User-agent: *\nAllow: /\n\nSitemap: ${sitemap.href}\n`, {
    headers: { 'Content-Type': 'text/plain; charset=utf-8' },
  });
};
