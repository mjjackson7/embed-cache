import { DiskStore } from './disk-store.ts';
import { toFloat32 } from './codec.ts';
import { embedKey } from './key.ts';
import { Lru } from './lru.ts';
import type { CacheStats, ComputeMany, ComputeOne, EmbedCacheOptions, VectorInput } from './types.ts';

interface Counters {
  memoryHits: number;
  diskHits: number;
  misses: number;
  computed: number;
  corrupt: number;
  diskWrites: number;
}

function freshCounters(): Counters {
  return { memoryHits: 0, diskHits: 0, misses: 0, computed: 0, corrupt: 0, diskWrites: 0 };
}

/**
 * In-memory LRU in front of an optional on-disk store, keyed by the SHA-256
 * of (namespace, model, text). See the README for the on-disk vector format.
 */
export class EmbedCache {
  readonly #memory: Lru<string, Float32Array>;
  readonly #disk: DiskStore | null;
  readonly #namespace: string;
  readonly #expectedDim: number | null;
  #counters = freshCounters();
  // Dedupes concurrent getOrCompute calls for the same key, so fifty parallel
  // callers hitting a cold cache still produce one call to `compute`.
  readonly #pending = new Map<string, Promise<Float32Array>>();

  constructor(options: EmbedCacheOptions = {}) {
    const { dir = null, maxEntries, maxBytes, namespace = '', expectedDim = null } = options;
    this.#memory = new Lru<string, Float32Array>({ maxEntries, maxBytes }, (vector) => vector.byteLength);
    this.#disk = dir == null ? null : new DiskStore(dir);
    this.#namespace = namespace;
    this.#expectedDim = expectedDim;
  }

  async get(model: string, text: string): Promise<Float32Array | undefined> {
    return this.#getByKey(embedKey(model, text, this.#namespace));
  }

  async set(model: string, text: string, vector: VectorInput): Promise<void> {
    await this.#setByKey(embedKey(model, text, this.#namespace), vector);
  }

  async delete(model: string, text: string): Promise<boolean> {
    const key = embedKey(model, text, this.#namespace);
    const inMemory = this.#memory.delete(key);
    const onDisk = this.#disk == null ? false : await this.#disk.delete(key);
    return inMemory || onDisk;
  }

  /**
   * Cached vector, or compute it once and store it. Concurrent calls for the
   * same (model, text) share one in-flight computation. If `compute` rejects,
   * nothing is cached, so the next call tries again.
   */
  async getOrCompute(model: string, text: string, compute: ComputeOne): Promise<Float32Array> {
    const key = embedKey(model, text, this.#namespace);
    const cached = await this.#getByKey(key);
    if (cached !== undefined) return cached;

    const pending = this.#pending.get(key);
    if (pending !== undefined) return pending;

    const promise = (async () => {
      try {
        const result = await compute(text, model);
        this.#counters.computed++;
        return await this.#setByKey(key, result);
      } finally {
        this.#pending.delete(key);
      }
    })();
    this.#pending.set(key, promise);
    return promise;
  }

  /**
   * As {@link getOrCompute}, but for many texts at once. `computeBatch` is
   * called at most once, with only the texts that were missing, deduplicated
   * and in first-seen order. The result lines up positionally with `texts`.
   */
  async getOrComputeMany(model: string, texts: string[], computeBatch: ComputeMany): Promise<Float32Array[]> {
    const keys = texts.map((text) => embedKey(model, text, this.#namespace));
    const found = new Map<string, Float32Array>();
    for (const key of new Set(keys)) {
      const vector = await this.#getByKey(key);
      if (vector !== undefined) found.set(key, vector);
    }

    const missingTexts: string[] = [];
    const missingKeys: string[] = [];
    const seenMissing = new Set<string>();
    for (let i = 0; i < texts.length; i++) {
      const key = keys[i]!;
      if (!found.has(key) && !seenMissing.has(key)) {
        seenMissing.add(key);
        missingTexts.push(texts[i]!);
        missingKeys.push(key);
      }
    }

    if (missingTexts.length > 0) {
      const computed = await computeBatch(missingTexts, model);
      if (computed.length !== missingTexts.length) {
        throw new RangeError(`computeBatch returned ${computed.length} vectors for ${missingTexts.length} texts`);
      }
      this.#counters.computed += computed.length;
      for (let i = 0; i < missingKeys.length; i++) {
        found.set(missingKeys[i]!, await this.#setByKey(missingKeys[i]!, computed[i]!));
      }
    }

    return keys.map((key) => found.get(key)!);
  }

  stats(): CacheStats {
    const lookups = this.#counters.memoryHits + this.#counters.diskHits + this.#counters.misses;
    const hitRate = lookups === 0 ? 0 : (this.#counters.memoryHits + this.#counters.diskHits) / lookups;
    return { ...this.#counters, entries: this.#memory.size, bytes: this.#memory.bytes, hitRate };
  }

  resetStats(): void {
    this.#counters = freshCounters();
  }

  clearMemory(): void {
    this.#memory.clear();
  }

  async clear(): Promise<void> {
    this.#memory.clear();
    if (this.#disk != null) await this.#disk.clear();
  }

  async #getByKey(key: string): Promise<Float32Array | undefined> {
    const cached = this.#memory.get(key);
    if (cached !== undefined) {
      this.#counters.memoryHits++;
      return cached;
    }
    if (this.#disk == null) {
      this.#counters.misses++;
      return undefined;
    }
    const { vector, corrupt } = await this.#disk.get(key);
    if (corrupt) this.#counters.corrupt++;
    if (vector === undefined) {
      this.#counters.misses++;
      return undefined;
    }
    this.#counters.diskHits++;
    this.#memory.set(key, vector);
    return vector;
  }

  async #setByKey(key: string, input: VectorInput): Promise<Float32Array> {
    const vector = toFloat32(input);
    if (this.#expectedDim != null && vector.length !== this.#expectedDim) {
      throw new RangeError(`expected a ${this.#expectedDim}-dimension vector, got ${vector.length}`);
    }
    this.#memory.set(key, vector);
    if (this.#disk != null) {
      await this.#disk.set(key, vector);
      this.#counters.diskWrites++;
    }
    return vector;
  }
}
