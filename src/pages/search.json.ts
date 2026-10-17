import { getPosts } from '../lib/posts';
import { href } from '../config';
export async function GET() {
  return Response.json((await getPosts()).map(p => ({ title: p.data.title, description: p.data.description, tags: p.data.tags, url: href(`posts/${p.id}/`), body: p.body || '' })));
}
