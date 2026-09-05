import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  DTYPE_FLOAT32,
  HEADER_BYTES,
  MAGIC,
  VECTOR_FORMAT_VERSION,
  decodeVector,
  encodeVector,
  encodedByteLength,
  toFloat32,
} from '../src/codec.ts';

test('encodedByteLength accounts for the header plus 4 bytes per dimension', () => {
  assert.equal(encodedByteLength(0), HEADER_BYTES);
  assert.equal(encodedByteLength(3), HEADER_BYTES + 12);
});

test('encode/decode round-trips a vector exactly', () => {
  const vector = Float32Array.from([1.5, -2.25, 0, 3.125]);
  const buffer = encodeVector(vector);
  assert.equal(buffer.length, encodedByteLength(vector.length));
  const decoded = decodeVector(buffer);
  assert.deepEqual(decoded, vector);
});

test('encode/decode round-trips an empty vector', () => {
  const vector = new Float32Array(0);
  const decoded = decodeVector(encodeVector(vector));
  assert.equal(decoded.length, 0);
});

test('encoded buffer carries the documented header fields', () => {
  const buffer = encodeVector(Float32Array.from([1, 2]));
  assert.equal(buffer.toString('ascii', 0, 4), MAGIC);
  assert.equal(buffer.readUInt8(4), VECTOR_FORMAT_VERSION);
  assert.equal(buffer.readUInt8(5), DTYPE_FLOAT32);
  assert.equal(buffer.readUInt32LE(8), 2);
});

test('decodeVector rejects a non-Buffer', () => {
  assert.throws(() => decodeVector(new Uint8Array(HEADER_BYTES) as unknown as Buffer), TypeError);
});

test('decodeVector rejects a buffer shorter than the header', () => {
  assert.throws(() => decodeVector(Buffer.alloc(HEADER_BYTES - 1)), /too short/);
});

test('decodeVector rejects a bad magic', () => {
  const buffer = encodeVector(Float32Array.from([1]));
  buffer.write('XXXX', 0, 'ascii');
  assert.throws(() => decodeVector(buffer), /bad magic/);
});

test('decodeVector rejects an unsupported format version', () => {
  const buffer = encodeVector(Float32Array.from([1]));
  buffer.writeUInt8(99, 4);
  assert.throws(() => decodeVector(buffer), /unsupported vector format version/);
});

test('decodeVector rejects an unsupported dtype', () => {
  const buffer = encodeVector(Float32Array.from([1]));
  buffer.writeUInt8(2, 5);
  assert.throws(() => decodeVector(buffer), /unsupported dtype/);
});

test('decodeVector rejects a truncated payload', () => {
  const buffer = encodeVector(Float32Array.from([1, 2, 3]));
  assert.throws(() => decodeVector(buffer.subarray(0, buffer.length - 4)), /truncated vector blob/);
});

test('decodeVector rejects a flipped payload bit as a checksum mismatch', () => {
  const buffer = encodeVector(Float32Array.from([1, 2, 3]));
  buffer[HEADER_BYTES] = buffer[HEADER_BYTES]! ^ 0xff;
  assert.throws(() => decodeVector(buffer), /checksum mismatch/);
});

test('toFloat32 copies a Float32Array rather than aliasing it', () => {
  const input = Float32Array.from([1, 2, 3]);
  const out = toFloat32(input);
  assert.deepEqual(out, input);
  input[0] = 99;
  assert.equal(out[0], 1);
});

test('toFloat32 converts a Float64Array and a plain number array', () => {
  assert.deepEqual(toFloat32(new Float64Array([1, 2])), Float32Array.from([1, 2]));
  assert.deepEqual(toFloat32([1, 2, 3]), Float32Array.from([1, 2, 3]));
});

test('toFloat32 rejects non-finite values', () => {
  assert.throws(() => toFloat32([1, NaN, 3]), RangeError);
  assert.throws(() => toFloat32([1, Infinity]), RangeError);
  assert.throws(() => toFloat32(Float32Array.from([1, -Infinity])), RangeError);
});

test('toFloat32 rejects unsupported input types', () => {
  assert.throws(() => toFloat32('nope' as never), TypeError);
});
