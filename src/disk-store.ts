import { randomBytes } from 'node:crypto';
import { mkdir, readFile, rename, rm, unlink, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { decodeVector, encodeVector } from './codec.ts';
import { keyToSegments } from './key.ts';

export interface DiskGetResult {
  /** The decoded vector, or undefined on a miss or a corrupt read. */
  vector: Float32Array | undefined;
  /** True if a file existed at this key but failed to decode, and was deleted. */
  corrupt: boolean;
}

/**
 * Sharded on-disk vector store: one file per key at
 * `<dir>/<key[0:2]>/<key[2:4]>/<key[4:]>.vec`, written via a temp file plus
 * rename so a crash mid-write can never leave a half-written file where a
 * reader would find it.
 */
export class DiskStore {
  readonly #dir: string;

  constructor(dir: string) {
    this.#dir = dir;
  }

  #pathFor(key: string): string {
    return join(this.#dir, ...keyToSegments(key));
  }

  async get(key: string): Promise<DiskGetResult> {
    let buffer: Buffer;
    try {
      buffer = await readFile(this.#pathFor(key));
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') return { vector: undefined, corrupt: false };
      throw err;
    }

    try {
      return { vector: decodeVector(buffer), corrupt: false };
    } catch {
      // Bad checksum, bad header, or truncated write from a crash before this
      // format existed -- either way the file cannot be trusted, so drop it
      // and let the caller recompute rather than serving garbage.
      await unlink(this.#pathFor(key)).catch(() => {});
      return { vector: undefined, corrupt: true };
    }
  }

  async set(key: string, vector: Float32Array): Promise<void> {
    const finalPath = this.#pathFor(key);
    const tmpPath = join(dirname(finalPath), `.tmp-${randomBytes(8).toString('hex')}`);
    await mkdir(dirname(finalPath), { recursive: true });
    await writeFile(tmpPath, encodeVector(vector));
    try {
      await rename(tmpPath, finalPath);
    } catch (err) {
      await unlink(tmpPath).catch(() => {});
      throw err;
    }
  }

  async delete(key: string): Promise<boolean> {
    try {
      await unlink(this.#pathFor(key));
      return true;
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') return false;
      throw err;
    }
  }

  /** Remove the entire store directory, including every shard. */
  async clear(): Promise<void> {
    await rm(this.#dir, { recursive: true, force: true });
  }
}
