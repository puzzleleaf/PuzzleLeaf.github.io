import type { APIContext } from 'astro';
import { getContent, type ContentItem } from '../lib/posts';
import { site, href } from '../config';

export async function GET(context: APIContext) {
  const content = (await getContent()).filter(item => !item.entry.data.sample);
  const url = (path: string) => new URL(href(path), context.site).href;
  const list = (items: ContentItem[]) => items.map(item => `- [${item.entry.data.title}](${new URL(item.url, context.site).href}): ${item.entry.data.description}\n  - [Markdown 원문](${new URL(item.rawUrl, context.site).href})`).join('\n');
  return new Response(`# ${site.title}

> ${site.description}

작성자: ${site.author}

## 탐색
- [소개](${url('about/')})
- [블로그](${url('')})
- [위키](${url('wiki/')})
- [태그](${url('tags/')})
- [RSS](${url('rss.xml')})
- [사이트맵](${url('sitemap.xml')})

## 블로그
${list(content.filter(item => item.kind === 'posts'))}

## 위키
${list(content.filter(item => item.kind === 'wiki'))}
`, { headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
}
