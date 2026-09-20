import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { isDeepStrictEqual } from 'node:util';
import { test } from 'node:test';
import { EmbedCache } from '../src/embed-cache.ts';

const DIM = 8;
const MODEL = 'fake-embed-8d';

// Same deterministic stand-in as examples/basic.ts, so this test asserts the
// exact numbers the README's transcript claims rather than just "it runs".
function fakeEmbedding(text: string, dim: number): number[] {
  const out = new Array<number>(dim);
  for (let i = 0; i < dim; i++) {
    let h = (2166136261 ^ i) >>> 0;
    for (let j = 0; j < text.length; j++) {
      h = Math.imul(h ^ text.charCodeAt(j), 16777619) >>> 0;
    }
    out[i] = (h % 2000) / 1000 - 1;
  }
  return out;
}

test('README transcript: batching, persistence, and stats end to end', async (t) => {
  const dir = await mkdtemp(join(tmpdir(), 'embed-cache-integration-'));
  t.after(() => rm(dir, { recursive: true, force: true }));

  const documents = ['Alpha document.', 'Beta document.', 'Gamma document.', 'Alpha document.'];

  let apiCalls = 0;
  const embedBatch = async (texts: string[]): Promise<number[][]> => {
    apiCalls++;
    return texts.map((text) => fakeEmbedding(text, DIM));
  };

  const cache = new EmbedCache({ dir, expectedDim: DIM });

  const firstPass = await cache.getOrComputeMany(MODEL, documents, embedBatch);
  assert.equal(firstPass.length, 4);
  assert.equal(apiCalls, 1); // one batch call for the three distinct texts

  const secondPass = await cache.getOrComputeMany(MODEL, documents, embedBatch);
  assert.equal(apiCalls, 1); // fully served from the memory tier, no new call
  assert.ok(isDeepStrictEqual(firstPass, secondPass));

  // A fresh instance over the same directory has an empty memory tier, so
  // this lookup can only be satisfied from disk.
  const restarted = new EmbedCache({ dir, expectedDim: DIM });
  const vector = await restarted.get(MODEL, documents[0]!);
  assert.equal(vector!.length, DIM);
  assert.deepEqual(vector, Float32Array.from(fakeEmbedding(documents[0]!, DIM)));
  assert.equal(restarted.stats().diskHits, 1);
  assert.equal(restarted.stats().misses, 0);

  const stats = cache.stats();
  // getOrComputeMany looks up each of the 3 distinct texts once per pass: all
  // 3 miss (and get computed) on the first pass, all 3 hit memory on the
  // second, matching the README's "hit rate 50.0%" line.
  assert.equal(stats.computed, 3);
  assert.equal(stats.misses, 3);
  assert.equal(stats.memoryHits, 3);
  assert.equal(stats.hitRate, 0.5);
  assert.equal(stats.entries, 3);
  assert.equal(stats.bytes, 3 * DIM * 4);
});
