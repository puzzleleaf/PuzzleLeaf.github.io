import { getWiki } from '../../../lib/posts';
import { site } from '../../../config';
import type { CollectionEntry } from 'astro:content';
import type { APIContext } from 'astro';
export async function getStaticPaths() {
  return (await getWiki()).map(post => ({ params: { id: post.id }, props: { post } }));
}
export function GET({ props }: APIContext) {
  const post = props.post as CollectionEntry<'wiki'>;
  return new Response(`# ${post.data.title}\n\n${post.data.description}\n\n작성자: ${site.author}\n작성일: ${post.data.date.toISOString().slice(0,10)}\n${post.data.updated ? `수정일: ${post.data.updated.toISOString().slice(0,10)}\n` : ''}\n${post.body || ''}`, { headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
}
