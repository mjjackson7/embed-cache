import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { isDeepStrictEqual } from 'node:util';
import { EmbedCache } from '../src/embed-cache.ts';

const DIM = 8;
const MODEL = 'fake-embed-8d';

/**
 * Stand-in for a real embedding call: same text always yields the same
 * vector, with no network involved, so this script runs offline.
 */
function fakeEmbedding(text: string, dim: number): number[] {
  const out = new Array<number>(dim);
  for (let i = 0; i < dim; i++) {
    let h = (2166136261 ^ i) >>> 0;
    for (let j = 0; j < text.length; j++) {
      h = Math.imul(h ^ text.charCodeAt(j), 16777619) >>> 0;
    }
    out[i] = (h % 2000) / 1000 - 1; // roughly [-1, 1)
  }
  return out;
}

async function main(): Promise<void> {
  const dir = await mkdtemp(join(tmpdir(), 'embed-cache-example-'));
  try {
    const documents = ['Alpha document.', 'Beta document.', 'Gamma document.', 'Alpha document.'];

    let apiCalls = 0;
    const embedBatch = async (texts: string[]): Promise<number[][]> => {
      apiCalls++;
      return texts.map((text) => fakeEmbedding(text, DIM));
    };

    const cache = new EmbedCache({ dir, expectedDim: DIM });

    const firstPass = await cache.getOrComputeMany(MODEL, documents, embedBatch);
    console.log(
      `first pass:  ${firstPass.length} vectors, ${apiCalls} api call(s) for ${new Set(documents).size} text(s)`,
    );

    const secondPass = await cache.getOrComputeMany(MODEL, documents, embedBatch);
    console.log(`second pass: ${secondPass.length} vectors, ${apiCalls} api call(s) total`);

    console.log(`identical:   ${isDeepStrictEqual(firstPass, secondPass)}`);

    // A fresh instance over the same directory has an empty memory tier, so
    // this lookup can only be satisfied from disk.
    const restarted = new EmbedCache({ dir, expectedDim: DIM });
    const vector = await restarted.get(MODEL, documents[0]!);
    console.log(`after restart: dim ${vector!.length}, stats ${JSON.stringify(restarted.stats())}`);

    const stats = cache.stats();
    console.log(
      `hit rate ${(stats.hitRate * 100).toFixed(1)}%  ` +
        `(memory ${stats.memoryHits}, disk ${stats.diskHits}, miss ${stats.misses}, computed ${stats.computed})  ` +
        `${stats.entries} entries / ${stats.bytes} bytes in memory`,
    );
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
