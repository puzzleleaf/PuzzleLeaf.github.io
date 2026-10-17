import type { APIContext } from 'astro';
import { href } from '../config';
export function GET(context: APIContext) {
  return new Response(`User-agent: *\nAllow: /\n\nUser-agent: OAI-SearchBot\nAllow: /\n\nSitemap: ${new URL(href('sitemap.xml'), context.site).href}\n`, { headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
}
