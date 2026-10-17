import { getPosts } from '../../lib/posts';
import type { CollectionEntry } from 'astro:content';
import type { APIContext } from 'astro';
export async function getStaticPaths() {
  return (await getPosts()).map(post => ({ params: { id: post.id }, props: { post } }));
}
export function GET({ props }: APIContext) {
  const post = props.post as CollectionEntry<'posts'>;
  return new Response(`# ${post.data.title}\n\n${post.data.description}\n\n작성자: puzzleleaf\n게시일: ${post.data.date.toISOString().slice(0,10)}\n${post.data.sample ? '\n> 블로그 기능을 보여주는 샘플 글입니다.\n' : ''}\n${post.body || ''}`, { headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
}
