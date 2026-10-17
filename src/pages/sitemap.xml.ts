import { getPosts } from '../lib/posts';
import { href } from '../config';
import { xml } from '../lib/xml';
import type { APIContext } from 'astro';
export async function GET(context: APIContext) {
  const posts = (await getPosts()).filter(p => !p.data.sample);
  const tags = [...new Set(posts.flatMap(p => p.data.tags))];
  const paths = [{ path: '' }, { path: 'about/' }, ...posts.map(p => ({ path: `posts/${p.id}/`, modified: (p.data.updated || p.data.date).toISOString() })), ...tags.map(t => ({ path: `tags/${encodeURIComponent(t)}/` }))];
  const body = `<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${paths.map(p => `<url><loc>${xml(new URL(href(p.path), context.site).href)}</loc>${'modified' in p ? `<lastmod>${p.modified}</lastmod>` : ''}</url>`).join('')}</urlset>`;
  return new Response(body, { headers: { 'Content-Type': 'application/xml; charset=utf-8' } });
}
