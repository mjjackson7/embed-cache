export { EmbedCache } from './embed-cache.ts';
export { Lru } from './lru.ts';
export type { LruOptions } from './lru.ts';
export { DiskStore } from './disk-store.ts';
export type { DiskGetResult } from './disk-store.ts';
export { embedKey, keyToSegments, isCacheKey, assertCacheKey, KEY_VERSION } from './key.ts';
export {
  encodeVector,
  decodeVector,
  toFloat32,
  encodedByteLength,
  MAGIC,
  HEADER_BYTES,
  VECTOR_FORMAT_VERSION,
  DTYPE_FLOAT32,
} from './codec.ts';
export type {
  CacheStats,
  ComputeMany,
  ComputeOne,
  EmbedCacheOptions,
  VectorInput,
} from './types.ts';
