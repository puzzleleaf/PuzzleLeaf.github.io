import sharp from 'sharp';
import { fileURLToPath } from 'node:url';

const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630">
  <rect width="1200" height="630" fill="#fff"/>
  <text x="100" y="140" fill="#777" font-size="22" font-family="Arial,sans-serif">TECHNICAL BLOG</text>
  <text x="95" y="325" fill="#242424" font-size="100" font-family="Arial,sans-serif" font-weight="700" letter-spacing="-4">puzzleleaf</text>
  <path d="M100 410h600" stroke="#ebebeb"/>
  <text x="100" y="493" fill="#777" font-size="26" font-family="Arial,sans-serif">blog.puzzleleaf.dev</text>
</svg>`;

const logo = await sharp(fileURLToPath(new URL('../src/assets/puzzleleaf-logo.png', import.meta.url)))
  .resize(280, 280).png().toBuffer();
await sharp(Buffer.from(svg)).composite([{ input: logo, left: 830, top: 175 }])
  .png().toFile(fileURLToPath(new URL('../public/og.png', import.meta.url)));
