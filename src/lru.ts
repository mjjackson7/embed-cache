export interface LruOptions {
  /** Max number of entries. Default: unlimited. */
  maxEntries?: number;
  /** Max total size, per the `sizeOf` function passed to the constructor. Default: unlimited. */
  maxBytes?: number;
}

/**
 * Least-recently-used cache that evicts on both entry count and a caller-defined
 * notion of size, not just count. EmbedCache uses it with `sizeOf` returning a
 * vector's byte length, since a handful of huge vectors can blow a memory budget
 * long before any entry-count limit would.
 *
 * A `Map` already preserves insertion order, so recency is tracked by deleting
 * and re-inserting a key on every touch rather than keeping a separate list.
 */
export class Lru<K, V> {
  readonly #maxEntries: number;
  readonly #maxBytes: number;
  readonly #sizeOf: (value: V) => number;
  readonly #entries = new Map<K, V>();
  #bytes = 0;

  constructor(options: LruOptions = {}, sizeOf: (value: V) => number = () => 0) {
    const { maxEntries = Infinity, maxBytes = Infinity } = options;
    if (maxEntries <= 0) throw new RangeError('maxEntries must be greater than zero');
    if (maxBytes <= 0) throw new RangeError('maxBytes must be greater than zero');
    this.#maxEntries = maxEntries;
    this.#maxBytes = maxBytes;
    this.#sizeOf = sizeOf;
  }

  get size(): number {
    return this.#entries.size;
  }

  get bytes(): number {
    return this.#bytes;
  }

  has(key: K): boolean {
    return this.#entries.has(key);
  }

  /** Look up a value, marking it most recently used. */
  get(key: K): V | undefined {
    const value = this.#entries.get(key);
    if (value === undefined) return undefined;
    this.#entries.delete(key);
    this.#entries.set(key, value);
    return value;
  }

  /** Insert or replace a value, then evict from the least-recently-used end until back under budget. */
  set(key: K, value: V): void {
    const existing = this.#entries.get(key);
    if (existing !== undefined) {
      this.#bytes -= this.#sizeOf(existing);
      this.#entries.delete(key);
    }
    this.#entries.set(key, value);
    this.#bytes += this.#sizeOf(value);
    this.#evict();
  }

  delete(key: K): boolean {
    const existing = this.#entries.get(key);
    if (existing === undefined) return false;
    this.#bytes -= this.#sizeOf(existing);
    this.#entries.delete(key);
    return true;
  }

  clear(): void {
    this.#entries.clear();
    this.#bytes = 0;
  }

  // Always leaves at least the entry just inserted, even if it alone exceeds
  // maxBytes -- otherwise a single oversized value would make set() a no-op.
  #evict(): void {
    while (this.#entries.size > 1 && (this.#entries.size > this.#maxEntries || this.#bytes > this.#maxBytes)) {
      const oldestKey = this.#entries.keys().next().value as K;
      const oldestValue = this.#entries.get(oldestKey)!;
      this.#entries.delete(oldestKey);
      this.#bytes -= this.#sizeOf(oldestValue);
    }
  }
}
