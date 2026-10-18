import { getCollection, type CollectionEntry } from 'astro:content';
import { href } from '../config';

export type ContentEntry = CollectionEntry<'posts'> | CollectionEntry<'wiki'>;
export type ContentItem = { entry: ContentEntry; kind: 'posts' | 'wiki'; url: string; rawUrl: string };

export async function getPosts() {
  return (await getCollection('posts', ({ data }) => !data.draft && data.date <= new Date()))
    .sort((a, b) => b.data.date.getTime() - a.data.date.getTime());
}

export async function getWiki() {
  return (await getCollection('wiki', ({ data }) => !data.draft && data.date <= new Date()))
    .sort((a, b) => a.data.order - b.data.order || a.data.title.localeCompare(b.data.title, 'ko'));
}

export async function getContent(): Promise<ContentItem[]> {
  const [posts, wiki] = await Promise.all([getPosts(), getWiki()]);
  return [
    ...posts.map(entry => ({ entry, kind: 'posts' as const, url: href(`posts/${entry.id}/`), rawUrl: href(`raw/${entry.id}.md`) })),
    ...wiki.map(entry => ({ entry, kind: 'wiki' as const, url: href(`wiki/${entry.id}/`), rawUrl: href(`raw/wiki/${entry.id}.md`) })),
  ];
}

export function readingTime(body = '') {
  const text = body
    .replace(/```[\s\S]*?```/g, '')
    .replace(/<svg\b[\s\S]*?<\/svg>/gi, '')
    .replace(/<[^>]+>/g, '')
    .replace(/!?\[([^\]]*)\]\([^)]+\)/g, '$1');
  return Math.max(1, Math.ceil(text.length / 550));
}
