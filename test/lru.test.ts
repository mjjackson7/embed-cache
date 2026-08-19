import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Lru } from '../src/lru.ts';

test('get/set round-trips and reports size', () => {
  const cache = new Lru<string, number>();
  assert.equal(cache.get('a'), undefined);
  cache.set('a', 1);
  cache.set('b', 2);
  assert.equal(cache.get('a'), 1);
  assert.equal(cache.get('b'), 2);
  assert.equal(cache.size, 2);
  assert.equal(cache.has('a'), true);
  assert.equal(cache.has('c'), false);
});

test('evicts the least recently used entry once maxEntries is exceeded', () => {
  const cache = new Lru<string, number>({ maxEntries: 2 });
  cache.set('a', 1);
  cache.set('b', 2);
  cache.get('a'); // touch a, so b becomes the oldest
  cache.set('c', 3);
  assert.equal(cache.has('a'), true);
  assert.equal(cache.has('b'), false);
  assert.equal(cache.has('c'), true);
  assert.equal(cache.size, 2);
});

test('evicts by size once maxBytes is exceeded', () => {
  const cache = new Lru<string, string>({ maxBytes: 10 }, (value) => value.length);
  cache.set('a', '12345'); // 5 bytes
  cache.set('b', '12345'); // 5 bytes, total 10, at budget
  assert.equal(cache.bytes, 10);
  cache.set('c', '123'); // 3 bytes, evicts a to make room
  assert.equal(cache.has('a'), false);
  assert.equal(cache.has('b'), true);
  assert.equal(cache.has('c'), true);
  assert.equal(cache.bytes, 8);
});

test('keeps a single oversized entry rather than refusing to cache it', () => {
  const cache = new Lru<string, string>({ maxBytes: 4 }, (value) => value.length);
  cache.set('a', '1234567890');
  assert.equal(cache.size, 1);
  assert.equal(cache.get('a'), '1234567890');
  assert.equal(cache.bytes, 10);
});

test('replacing a key updates its byte accounting instead of double-counting', () => {
  const cache = new Lru<string, string>({}, (value) => value.length);
  cache.set('a', '12345');
  cache.set('a', '12');
  assert.equal(cache.bytes, 2);
  assert.equal(cache.size, 1);
});

test('delete removes an entry and its bytes, and reports whether it existed', () => {
  const cache = new Lru<string, string>({}, (value) => value.length);
  cache.set('a', '12345');
  assert.equal(cache.delete('a'), true);
  assert.equal(cache.delete('a'), false);
  assert.equal(cache.size, 0);
  assert.equal(cache.bytes, 0);
});

test('clear drops every entry and resets the byte count', () => {
  const cache = new Lru<string, string>({}, (value) => value.length);
  cache.set('a', '123');
  cache.set('b', '456');
  cache.clear();
  assert.equal(cache.size, 0);
  assert.equal(cache.bytes, 0);
  assert.equal(cache.has('a'), false);
});

test('rejects non-positive limits', () => {
  assert.throws(() => new Lru({ maxEntries: 0 }), RangeError);
  assert.throws(() => new Lru({ maxBytes: -1 }), RangeError);
});
