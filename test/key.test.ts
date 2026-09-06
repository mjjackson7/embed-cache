import assert from 'node:assert/strict';
import { test } from 'node:test';
import { assertCacheKey, embedKey, isCacheKey, keyToSegments } from '../src/key.ts';

test('embedKey returns a 64-char lowercase hex digest', () => {
  const key = embedKey('text-embedding-3-small', 'hello world');
  assert.equal(key.length, 64);
  assert.match(key, /^[0-9a-f]{64}$/);
});

test('embedKey is deterministic for the same inputs', () => {
  const a = embedKey('model', 'text', 'ns');
  const b = embedKey('model', 'text', 'ns');
  assert.equal(a, b);
});

test('embedKey defaults namespace to the empty string', () => {
  assert.equal(embedKey('model', 'text'), embedKey('model', 'text', ''));
});

test('embedKey changes with model, text, or namespace', () => {
  const base = embedKey('model-a', 'text', 'ns');
  assert.notEqual(base, embedKey('model-b', 'text', 'ns'));
  assert.notEqual(base, embedKey('model-a', 'other', 'ns'));
  assert.notEqual(base, embedKey('model-a', 'text', 'other-ns'));
});

test('embedKey framing prevents field-boundary collisions', () => {
  // Without length-prefixing, ('a', 'b|c') and ('a|b', 'c') would hash the same.
  const a = embedKey('a', 'b|c');
  const b = embedKey('a|b', 'c');
  assert.notEqual(a, b);
});

test('embedKey is sensitive to whitespace and casing in the text', () => {
  const base = embedKey('model', 'Hello World');
  assert.notEqual(base, embedKey('model', 'hello world'));
  assert.notEqual(base, embedKey('model', 'Hello World '));
});

test('embedKey rejects an empty model', () => {
  assert.throws(() => embedKey('', 'text'), TypeError);
});

test('embedKey rejects non-string arguments', () => {
  assert.throws(() => embedKey(1 as unknown as string, 'text'), TypeError);
  assert.throws(() => embedKey('model', 1 as unknown as string), TypeError);
  assert.throws(() => embedKey('model', 'text', 1 as unknown as string), TypeError);
});

test('isCacheKey accepts a well-formed key and rejects malformed ones', () => {
  const key = embedKey('model', 'text');
  assert.equal(isCacheKey(key), true);
  assert.equal(isCacheKey(key.toUpperCase()), false);
  assert.equal(isCacheKey(key.slice(0, -1)), false);
  assert.equal(isCacheKey(`${key}0`), false);
  assert.equal(isCacheKey('not-a-key'), false);
  assert.equal(isCacheKey(''), false);
});

test('assertCacheKey passes through a valid key silently', () => {
  const key = embedKey('model', 'text');
  assert.doesNotThrow(() => assertCacheKey(key));
});

test('assertCacheKey throws on an invalid key', () => {
  assert.throws(() => assertCacheKey('nope'), TypeError);
});

test('keyToSegments splits into two 2-char directories and a .vec file', () => {
  const key = embedKey('model', 'text');
  const segments = keyToSegments(key);
  assert.equal(segments.length, 3);
  assert.equal(segments[0], key.slice(0, 2));
  assert.equal(segments[1], key.slice(2, 4));
  assert.equal(segments[2], `${key.slice(4)}.vec`);
  assert.equal(segments.join('/'), `${key.slice(0, 2)}/${key.slice(2, 4)}/${key.slice(4)}.vec`);
});

test('keyToSegments rejects a value that is not a cache key', () => {
  assert.throws(() => keyToSegments('short'), TypeError);
});
