import type { DistanceMetric } from './types.js';

export type DistanceFn = (a: Float32Array, b: Float32Array) => number;

export function dot(a: Float32Array, b: Float32Array): number {
  const n = a.length;
  let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  let i = 0;
  const n4 = n - (n % 4);
  for (; i < n4; i += 4) {
    s0 += a[i] * b[i];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
  }
  for (; i < n; i++) s0 += a[i] * b[i];
  return s0 + s1 + s2 + s3;
}

/** Cosine distance for vectors that are already unit length (what cosine collections store). */
export const normalizedCosineDistance: DistanceFn = (a, b) => {
  const d = 1 - dot(a, b);
  return d < 0 ? 0 : d;
};

export function norm(a: Float32Array): number {
  return Math.sqrt(dot(a, a));
}

export function normalize(a: Float32Array): Float32Array {
  const n = norm(a);
  if (n === 0) return a;
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] / n;
  return out;
}

export const cosineDistance: DistanceFn = (a, b) => {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 1;
  const c = d / Math.sqrt(na * nb);
  return 1 - Math.max(-1, Math.min(1, c));
};

export const euclideanDistance: DistanceFn = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
};

export const dotProductDistance: DistanceFn = (a, b) => -dot(a, b);

export const manhattanDistance: DistanceFn = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
  return s;
};

export function distanceFor(metric: DistanceMetric): DistanceFn {
  switch (metric) {
    case 'cosine':
      return cosineDistance;
    case 'euclidean':
      return euclideanDistance;
    case 'dotProduct':
      return dotProductDistance;
    case 'manhattan':
      return manhattanDistance;
  }
}

/** Derived similarity, higher is better. Documented on every surface because callers routinely invert it. */
export function similarityFor(metric: DistanceMetric, distance: number): number {
  switch (metric) {
    case 'cosine':
      return 1 - distance;
    case 'dotProduct':
      return -distance;
    case 'euclidean':
    case 'manhattan':
      return 1 / (1 + distance);
  }
}

export function toFloat32(v: Float32Array | number[]): Float32Array {
  return v instanceof Float32Array ? v : Float32Array.from(v);
}

export function f32ToBase64(v: Float32Array): string {
  return Buffer.from(v.buffer, v.byteOffset, v.byteLength).toString('base64');
}

export function base64ToF32(s: string): Float32Array {
  const b = Buffer.from(s, 'base64');
  const copy = new Uint8Array(b.byteLength);
  copy.set(b);
  return new Float32Array(copy.buffer);
}
