import { endianness } from 'node:os';
import { crc32 } from 'node:zlib';
import type { VectorInput } from './types.ts';

/**
 * On-disk layout of one vector:
 *
 *   offset  size  field
 *   0       4     magic "EMBC"
 *   4       1     format version
 *   5       1     dtype (1 = float32)
 *   6       2     reserved, zero
 *   8       4     dim, uint32 little-endian
 *   12      4     crc32 of the payload, uint32 little-endian
 *   16      4*dim payload, float32 little-endian
 *
 * JSON would cost roughly 5x the bytes and lose the exact float32 bits, which
 * is why this exists at all.
 */
export const MAGIC = 'EMBC';
export const HEADER_BYTES = 16;
export const VECTOR_FORMAT_VERSION = 1;
export const DTYPE_FLOAT32 = 1;

const HOST_IS_LITTLE_ENDIAN = endianness() === 'LE';

export function encodedByteLength(dim: number): number {
  return HEADER_BYTES + dim * 4;
}

/** Copy any numeric sequence into a Float32Array, rejecting unusable values. */
export function toFloat32(input: VectorInput): Float32Array {
  if (input instanceof Float32Array) {
    assertFinite(input);
    // Copy so later mutation of the caller's array cannot corrupt the cache.
    return Float32Array.from(input);
  }
  if (input instanceof Float64Array || Array.isArray(input)) {
    const out = Float32Array.from(input as ArrayLike<number>);
    assertFinite(out, input as ArrayLike<number>);
    return out;
  }
  throw new TypeError('vector must be a Float32Array, a Float64Array or an array of numbers');
}

export function encodeVector(vector: Float32Array): Buffer {
  const buffer = Buffer.allocUnsafe(encodedByteLength(vector.length));
  const payload = buffer.subarray(HEADER_BYTES);

  if (HOST_IS_LITTLE_ENDIAN) {
    Buffer.from(vector.buffer, vector.byteOffset, vector.byteLength).copy(payload);
  } else {
    for (let i = 0; i < vector.length; i++) payload.writeFloatLE(vector[i]!, i * 4);
  }

  buffer.write(MAGIC, 0, 'ascii');
  buffer.writeUInt8(VECTOR_FORMAT_VERSION, 4);
  buffer.writeUInt8(DTYPE_FLOAT32, 5);
  buffer.writeUInt16LE(0, 6);
  buffer.writeUInt32LE(vector.length, 8);
  buffer.writeUInt32LE(crc32(payload) >>> 0, 12);
  return buffer;
}

/** Inverse of {@link encodeVector}. Throws on anything it cannot trust. */
export function decodeVector(buffer: Buffer): Float32Array {
  if (!Buffer.isBuffer(buffer)) throw new TypeError('expected a Buffer');
  if (buffer.length < HEADER_BYTES) {
    throw new Error(`vector blob is too short: ${buffer.length} bytes, need at least ${HEADER_BYTES}`);
  }
  const magic = buffer.toString('ascii', 0, 4);
  if (magic !== MAGIC) throw new Error(`bad magic ${JSON.stringify(magic)}: not a vector blob`);

  const version = buffer.readUInt8(4);
  if (version !== VECTOR_FORMAT_VERSION) throw new Error(`unsupported vector format version ${version}`);

  const dtype = buffer.readUInt8(5);
  if (dtype !== DTYPE_FLOAT32) throw new Error(`unsupported dtype ${dtype}`);

  const dim = buffer.readUInt32LE(8);
  const expected = encodedByteLength(dim);
  if (buffer.length !== expected) {
    throw new Error(`truncated vector blob: dim ${dim} needs ${expected} bytes, got ${buffer.length}`);
  }

  const payload = buffer.subarray(HEADER_BYTES);
  const stored = buffer.readUInt32LE(12);
  const actual = crc32(payload) >>> 0;
  if (stored !== actual) {
    throw new Error(`checksum mismatch: stored ${stored}, computed ${actual}`);
  }

  const out = new Float32Array(dim);
  for (let i = 0; i < dim; i++) out[i] = payload.readFloatLE(i * 4);
  return out;
}

function assertFinite(values: Float32Array, source?: ArrayLike<number>): void {
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i]!)) {
      const original = source ? source[i] : values[i];
      throw new RangeError(`vector[${i}] is not a finite number: ${String(original)}`);
    }
  }
}
