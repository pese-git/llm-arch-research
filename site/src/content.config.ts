import { defineCollection } from 'astro:content';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';

// Страницы генерирует scripts/sync-docs.mjs из ../docs
export const collections = {
  docs: defineCollection({ loader: docsLoader(), schema: docsSchema() }),
};
