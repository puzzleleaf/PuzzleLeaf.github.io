import sharp from 'sharp';
import { writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const source = fileURLToPath(new URL('../src/assets/puzzleleaf-logo.png', import.meta.url));
const output = name => fileURLToPath(new URL(`../public/${name}`, import.meta.url));

// Keep the supplied artwork and alpha channel; only resize for each display size.
await sharp(source).resize(512, 512).webp({ lossless: true }).toFile(output('logo.webp'));
await sharp(source).resize(32, 32).png().toFile(output('favicon-32.png'));
await sharp(source).resize(180, 180).png().toFile(output('apple-touch-icon.png'));
const icon = await sharp(source).resize(128, 128).png().toBuffer();
await writeFile(output('favicon.svg'), `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128"><image width="128" height="128" href="data:image/png;base64,${icon.toString('base64')}"/></svg>\n`);
