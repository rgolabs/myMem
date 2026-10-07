import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

export function tmpDir(name: string): string {
  const d = fs.mkdtempSync(path.join(os.tmpdir(), `mem-${name}-`));
  return d;
}

export function rng(seed = 1) {
  let a = seed >>> 0;
  return () => {
    a = (a * 1664525 + 1013904223) >>> 0;
    return a / 4294967296;
  };
}

export function randomVector(dim: number, r: () => number): Float32Array {
  const v = new Float32Array(dim);
  for (let i = 0; i < dim; i++) v[i] = r() - 0.5;
  return v;
}
