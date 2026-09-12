import assert from 'node:assert/strict';
import { mkdtemp, readdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test, type TestContext } from 'node:test';
import { encodeVector } from '../src/codec.ts';
import { DiskStore } from '../src/disk-store.ts';
import { embedKey, keyToSegments } from '../src/key.ts';

async function tempDir(t: TestContext): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), 'disk-store-test-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  return dir;
}

test('get on an empty store is a miss, not an error', async (t) => {
  const store = new DiskStore(await tempDir(t));
  const result = await store.get(embedKey('model', 'hello'));
  assert.deepEqual(result, { vector: undefined, corrupt: false });
});

test('set then get round-trips the vector', async (t) => {
  const store = new DiskStore(await tempDir(t));
  const key = embedKey('model', 'hello');
  await store.set(key, Float32Array.from([1, 2, 3]));
  const result = await store.get(key);
  assert.deepEqual(result.vector, Float32Array.from([1, 2, 3]));
  assert.equal(result.corrupt, false);
});

test('set overwrites a previous value at the same key', async (t) => {
  const store = new DiskStore(await tempDir(t));
  const key = embedKey('model', 'hello');
  await store.set(key, Float32Array.from([1, 2]));
  await store.set(key, Float32Array.from([9, 9, 9]));
  assert.deepEqual((await store.get(key)).vector, Float32Array.from([9, 9, 9]));
});

test('files land at the sharded path keyToSegments describes', async (t) => {
  const dir = await tempDir(t);
  const store = new DiskStore(dir);
  const key = embedKey('model', 'hello');
  await store.set(key, Float32Array.from([1]));

  const segments = keyToSegments(key);
  const info = await stat(join(dir, ...segments));
  assert.equal(info.isFile(), true);
});

test('set leaves no temp files behind', async (t) => {
  const dir = await tempDir(t);
  const store = new DiskStore(dir);
  const key = embedKey('model', 'hello');
  await store.set(key, Float32Array.from([1, 2, 3]));

  const [shard1] = keyToSegments(key);
  const entries = await readdir(join(dir, shard1));
  assert.equal(entries.some((name) => name.startsWith('.tmp-')), false);
});

test('delete removes an existing entry and reports it', async (t) => {
  const store = new DiskStore(await tempDir(t));
  const key = embedKey('model', 'hello');
  await store.set(key, Float32Array.from([1]));
  assert.equal(await store.delete(key), true);
  assert.deepEqual(await store.get(key), { vector: undefined, corrupt: false });
});

test('delete on a missing entry returns false rather than throwing', async (t) => {
  const store = new DiskStore(await tempDir(t));
  assert.equal(await store.delete(embedKey('model', 'hello')), false);
});

test('a corrupt file is reported and removed on read', async (t) => {
  const dir = await tempDir(t);
  const store = new DiskStore(dir);
  const key = embedKey('model', 'hello');

  const good = encodeVector(Float32Array.from([1, 2, 3]));
  const corrupted = Buffer.from(good);
  corrupted[corrupted.length - 1] ^= 0xff; // flip a payload bit so the crc32 no longer matches
  const path = join(dir, ...keyToSegments(key));
  await store.set(key, Float32Array.from([1, 2, 3]));
  await writeFile(path, corrupted);

  const result = await store.get(key);
  assert.equal(result.vector, undefined);
  assert.equal(result.corrupt, true);

  // the bad file must be gone so a later set() is not blocked, and a later get() is a clean miss
  await assert.rejects(readFile(path));
  assert.deepEqual(await store.get(key), { vector: undefined, corrupt: false });
});

test('clear removes every shard and the store keeps working afterward', async (t) => {
  const dir = await tempDir(t);
  const store = new DiskStore(dir);
  const keyA = embedKey('model', 'a');
  const keyB = embedKey('model', 'b');
  await store.set(keyA, Float32Array.from([1]));
  await store.set(keyB, Float32Array.from([2]));

  await store.clear();

  assert.deepEqual(await store.get(keyA), { vector: undefined, corrupt: false });
  assert.deepEqual(await store.get(keyB), { vector: undefined, corrupt: false });

  await store.set(keyA, Float32Array.from([7]));
  assert.deepEqual((await store.get(keyA)).vector, Float32Array.from([7]));
});
