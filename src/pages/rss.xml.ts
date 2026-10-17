import { getPosts } from '../lib/posts';
import { site, href } from '../config';
import { xml } from '../lib/xml';
import type { APIContext } from 'astro';
export async function GET(context: APIContext) {
  const posts = (await getPosts()).filter(p => !p.data.sample);
  const url = (path: string) => new URL(href(path), context.site).href;
  const body = `<?xml version="1.0" encoding="UTF-8"?><rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom" xmlns:dc="http://purl.org/dc/elements/1.1/"><channel><title>${xml(site.title)}</title><link>${xml(url(''))}</link><description>${xml(site.description)}</description><language>ko-kr</language><atom:link href="${xml(url('rss.xml'))}" rel="self" type="application/rss+xml"/>${posts.map(p => `<item><title>${xml(p.data.title)}</title><link>${xml(url(`posts/${p.id}/`))}</link><guid isPermaLink="true">${xml(url(`posts/${p.id}/`))}</guid><description>${xml(p.data.description)}</description><pubDate>${p.data.date.toUTCString()}</pubDate><dc:creator>${xml(site.author)}</dc:creator></item>`).join('')}</channel></rss>`;
  return new Response(body, { headers: { 'Content-Type': 'application/rss+xml; charset=utf-8' } });
}
