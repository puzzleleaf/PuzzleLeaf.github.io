import type { APIContext } from 'astro';
import { getPosts } from '../lib/posts';
import { site, href } from '../config';
export async function GET(context: APIContext) {
  const posts = (await getPosts()).filter(p => !p.data.sample);
  const url = (path: string) => new URL(href(path), context.site).href;
  return new Response(`# ${site.title}\n\n> ${site.description}\n\n한국어 개발 블로그. 작성자는 ${site.author}입니다. 각 글의 게시일과 수정일은 해당 페이지에서 확인할 수 있습니다.\n\n## 소개\n- [작성자 소개](${url('about/')})\n- [RSS](${url('rss.xml')})\n- [사이트맵](${url('sitemap.xml')})\n\n## 공개 글\n${posts.map(p => `- [${p.data.title}](${url(`posts/${p.id}/`)}): ${p.data.description}\n  - [Markdown 원문](${url(`raw/${p.id}.md`)})`).join('\n')}\n\n이 파일은 보조 탐색 문서이며 검색 노출이나 AI 인용을 보장하지 않습니다. 샘플 글은 이 목록에서 제외됩니다.\n`, { headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
}
