// Картинка превью ссылок: src/assets/og.svg → public/og.png (1200×630).
// PNG хранится в git: шрифты берутся из системы, и на другой машине текст может
// отрисоваться иначе. Пересобрать после правки og.svg: npm run og-image.
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';

const siteDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const out = path.join(siteDir, 'public/og.png');
await sharp(path.join(siteDir, 'src/assets/og.svg')).png({ compressionLevel: 9 }).toFile(out);
console.log(`[og-image] ${path.relative(siteDir, out)}`);
