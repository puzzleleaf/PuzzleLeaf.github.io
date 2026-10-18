import { getContent } from '../lib/posts';
export async function GET() {
  return Response.json((await getContent()).map(({ entry, kind, url }) => ({ title: entry.data.title, description: entry.data.description, kind: kind === 'wiki' ? '위키' : '블로그', tags: entry.data.tags, url, body: `${entry.data.summary.join(' ')} ${entry.body || ''}` })));
}
