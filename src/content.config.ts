import { defineCollection } from 'astro:content';
import { z } from 'astro/zod';
import { glob } from 'astro/loaders';

const entrySchema = z.object({
    title: z.string(),
    description: z.string(),
    summary: z.array(z.string()).max(8).default([]),
    date: z.coerce.date(),
    updated: z.coerce.date().optional(),
    category: z.string(),
    tags: z.array(z.string()).default([]),
    featured: z.boolean().default(false),
    draft: z.boolean().default(false),
    sample: z.boolean().default(false),
    interactive: z.array(z.enum(['pi-harness'])).default([]),
    related: z.array(z.string()).default([]),
    order: z.number().default(0),
});

const posts = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/posts' }),
  schema: entrySchema,
});

const wiki = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/wiki' }),
  schema: entrySchema,
});

export const collections = { posts, wiki };
