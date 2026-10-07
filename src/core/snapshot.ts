/**
 * Snapshot file format: gzip( JSON { version, createdAt, kind, checksum, body } ) where
 * checksum = sha256(JSON.stringify(body)). Restore verifies the checksum before touching disk.
 */
import fs from 'node:fs';
import zlib from 'node:zlib';
import { MemError } from './errors.js';
import { sha256 } from './witness.js';
import { writeFileAtomic } from './fsutil.js';

export interface SnapshotFile<T> {
  version: number;
  createdAt: string;
  kind: string;
  checksum: string;
  body: T;
}

export function writeSnapshot<T>(file: string, kind: string, body: T): { bytes: number; checksum: string } {
  const bodyStr = JSON.stringify(body);
  const checksum = sha256(bodyStr);
  const doc = `{"version":1,"createdAt":${JSON.stringify(new Date().toISOString())},"kind":${JSON.stringify(kind)},"checksum":"${checksum}","body":${bodyStr}}`;
  const gz = zlib.gzipSync(Buffer.from(doc, 'utf8'), { level: 6 });
  writeFileAtomic(file, gz);
  return { bytes: gz.byteLength, checksum };
}

export function readSnapshot<T>(file: string, expectedKind?: string): SnapshotFile<T> {
  let raw: Buffer;
  try {
    raw = fs.readFileSync(file);
  } catch (e: any) {
    throw new MemError('NOT_FOUND', `snapshot ${file}: ${e?.message ?? e}`);
  }
  let doc: SnapshotFile<T>;
  try {
    const text = zlib.gunzipSync(raw).toString('utf8');
    doc = JSON.parse(text);
    // verify against the canonical serialisation of the body
    if (sha256(JSON.stringify(doc.body)) !== doc.checksum) throw new Error('checksum mismatch');
  } catch (e: any) {
    throw new MemError('CORRUPT', `snapshot ${file} is corrupt: ${e?.message ?? e}`);
  }
  if (expectedKind && doc.kind !== expectedKind) throw new MemError('INVALID_ARGUMENT', `snapshot ${file} is of kind ${doc.kind}, expected ${expectedKind}`);
  return doc;
}
