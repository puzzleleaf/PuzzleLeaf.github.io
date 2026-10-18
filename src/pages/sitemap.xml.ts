import { getContent } from '../lib/posts';
import { href } from '../config';
import { xml } from '../lib/xml';
import type { APIContext } from 'astro';
export async function GET(context: APIContext) {
  const posts = (await getContent()).filter(p => !p.entry.data.sample);
  const tags = [...new Set(posts.flatMap(p => p.entry.data.tags))];
  const paths = [{ url: href('') }, { url: href('about/') }, { url: href('tags/') }, { url: href('wiki/') }, ...posts.map(p => ({ url: p.url, modified: (p.entry.data.updated || p.entry.data.date).toISOString() })), ...tags.map(t => ({ url: href(`tags/${encodeURIComponent(t)}/`) }))];
  const body = `<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${paths.map(p => `<url><loc>${xml(new URL(p.url, context.site).href)}</loc>${'modified' in p ? `<lastmod>${p.modified}</lastmod>` : ''}</url>`).join('')}</urlset>`;
  return new Response(body, { headers: { 'Content-Type': 'application/xml; charset=utf-8' } });
}
