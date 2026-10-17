import { defineConfig } from 'astro/config';

export default defineConfig({
  site: process.env.SITE_URL || 'https://blog.puzzleleaf.dev',
  base: process.env.BASE_PATH || '/',
  trailingSlash: 'always',
  markdown: { shikiConfig: { theme: 'github-dark' } },
  devToolbar: { enabled: false },
});
