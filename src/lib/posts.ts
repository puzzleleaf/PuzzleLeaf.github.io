import { getCollection } from 'astro:content';

export async function getPosts() {
  return (await getCollection('posts', ({ data }) => !data.draft && data.date <= new Date()))
    .sort((a, b) => b.data.date.getTime() - a.data.date.getTime());
}

export function readingTime(body = '') {
  return Math.max(1, Math.ceil(body.replace(/```[\s\S]*?```/g, '').length / 550));
}
