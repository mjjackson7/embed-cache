import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { EmbedCache } from '../src/embed-cache.ts';

test('get/set/delete round-trip in a memory-only cache', async () => {
  const cache = new EmbedCache();
  assert.equal(await cache.get('model-a', 'hello'), undefined);
  await cache.set('model-a', 'hello', [1, 2, 3]);
  assert.deepEqual(await cache.get('model-a', 'hello'), Float32Array.from([1, 2, 3]));
  assert.equal(await cache.delete('model-a', 'hello'), true);
  assert.equal(await cache.delete('model-a', 'hello'), false);
});

test('different models or namespaces do not collide on the same text', async () => {
  const cache = new EmbedCache({ namespace: 'ns' });
  await cache.set('model-a', 'hello', [1]);
  await cache.set('model-b', 'hello', [2]);
  assert.deepEqual(await cache.get('model-a', 'hello'), Float32Array.from([1]));
  assert.deepEqual(await cache.get('model-b', 'hello'), Float32Array.from([2]));
});

test('getOrCompute caches the result and does not call compute again', async () => {
  const cache = new EmbedCache();
  let calls = 0;
  const compute = async () => {
    calls++;
    return [1, 2, 3];
  };
  const first = await cache.getOrCompute('model-a', 'hello', compute);
  const second = await cache.getOrCompute('model-a', 'hello', compute);
  assert.equal(calls, 1);
  assert.deepEqual(first, second);
});

test('getOrCompute shares one in-flight computation across concurrent callers', async () => {
  const cache = new EmbedCache();
  let calls = 0;
  const compute = async () => {
    calls++;
    await new Promise((resolve) => setTimeout(resolve, 5));
    return [4, 5, 6];
  };
  const [a, b, c] = await Promise.all([
    cache.getOrCompute('model-a', 'hello', compute),
    cache.getOrCompute('model-a', 'hello', compute),
    cache.getOrCompute('model-a', 'hello', compute),
  ]);
  assert.equal(calls, 1);
  assert.deepEqual(a, Float32Array.from([4, 5, 6]));
  assert.deepEqual(b, Float32Array.from([4, 5, 6]));
  assert.deepEqual(c, Float32Array.from([4, 5, 6]));
});

test('getOrCompute does not cache a rejected computation', async () => {
  const cache = new EmbedCache();
  await assert.rejects(cache.getOrCompute('model-a', 'hello', async () => {
    throw new Error('boom');
  }));
  assert.equal(await cache.get('model-a', 'hello'), undefined);

  const recovered = await cache.getOrCompute('model-a', 'hello', async () => [9]);
  assert.deepEqual(recovered, Float32Array.from([9]));
});

test('getOrComputeMany calls computeBatch once with only the missing, deduplicated texts', async () => {
  const cache = new EmbedCache();
  await cache.set('model-a', 'b', [20]);

  const seen: string[][] = [];
  const vectors = await cache.getOrComputeMany('model-a', ['a', 'b', 'c', 'a', 'c'], async (missing) => {
    seen.push(missing);
    return missing.map((text) => [text.charCodeAt(0)]);
  });

  assert.deepEqual(seen, [['a', 'c']]);
  assert.deepEqual(
    vectors,
    ['a', 'b', 'c', 'a', 'c'].map((text) => (text === 'b' ? Float32Array.from([20]) : Float32Array.from([text.charCodeAt(0)]))),
  );
});

test('getOrComputeMany skips computeBatch entirely when nothing is missing', async () => {
  const cache = new EmbedCache();
  await cache.set('model-a', 'a', [1]);
  let called = false;
  const vectors = await cache.getOrComputeMany('model-a', ['a', 'a'], async () => {
    called = true;
    return [];
  });
  assert.equal(called, false);
  assert.deepEqual(vectors, [Float32Array.from([1]), Float32Array.from([1])]);
});

test('getOrComputeMany rejects when computeBatch returns the wrong number of vectors', async () => {
  const cache = new EmbedCache();
  await assert.rejects(
    cache.getOrComputeMany('model-a', ['a', 'b'], async () => [[1]]),
    RangeError,
  );
});

test('expectedDim rejects vectors of the wrong length', async () => {
  const cache = new EmbedCache({ expectedDim: 3 });
  await assert.rejects(cache.set('model-a', 'hello', [1, 2]), RangeError);
  await assert.rejects(cache.getOrCompute('model-a', 'hello', async () => [1, 2]), RangeError);
});

test('stats tracks hits, misses and hit rate; resetStats zeroes them', async () => {
  const cache = new EmbedCache();
  await cache.get('model-a', 'miss');
  await cache.set('model-a', 'hit', [1]);
  await cache.get('model-a', 'hit');

  const stats = cache.stats();
  assert.equal(stats.misses, 1);
  assert.equal(stats.memoryHits, 1);
  assert.equal(stats.entries, 1);
  assert.equal(stats.hitRate, 0.5);

  cache.resetStats();
  const reset = cache.stats();
  assert.equal(reset.misses, 0);
  assert.equal(reset.memoryHits, 0);
  assert.equal(reset.hitRate, 0);
  assert.equal(reset.entries, 1); // entries reflects current memory state, not a counter
});

test('clearMemory empties the in-memory tier without touching stats counters', async () => {
  const cache = new EmbedCache();
  await cache.set('model-a', 'hello', [1]);
  cache.clearMemory();
  assert.equal(await cache.get('model-a', 'hello'), undefined);
  assert.equal(cache.stats().entries, 0);
});

test('persists to disk and survives a fresh cache instance over the same directory', async (t) => {
  const dir = await mkdtemp(join(tmpdir(), 'embed-cache-test-'));
  t.after(() => rm(dir, { recursive: true, force: true }));

  const first = new EmbedCache({ dir });
  await first.set('model-a', 'hello', [1, 2, 3, 4]);
  first.clearMemory();

  const second = new EmbedCache({ dir });
  const vector = await second.get('model-a', 'hello');
  assert.deepEqual(vector, Float32Array.from([1, 2, 3, 4]));
  assert.equal(second.stats().diskHits, 1);
});

test('clear removes both the memory and disk tiers', async (t) => {
  const dir = await mkdtemp(join(tmpdir(), 'embed-cache-test-'));
  t.after(() => rm(dir, { recursive: true, force: true }));

  const cache = new EmbedCache({ dir });
  await cache.set('model-a', 'hello', [1, 2, 3]);
  await cache.clear();

  assert.equal(await cache.get('model-a', 'hello'), undefined);
  assert.equal(cache.stats().entries, 0);

  const fresh = new EmbedCache({ dir });
  assert.equal(await fresh.get('model-a', 'hello'), undefined);
});
