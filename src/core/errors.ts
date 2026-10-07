/**
 * Typed errors. Every boundary violation (dimension, id length, metadata size,
 * embedding-space mismatch, unsupported query construct) fails with one of these
 * rather than silently truncating or returning an empty result.
 */
export type MemErrorCode =
  | 'DIMENSION_MISMATCH'
  | 'EMBEDDING_SPACE_MISMATCH'
  | 'INVALID_ARGUMENT'
  | 'LIMIT_EXCEEDED'
  | 'NOT_FOUND'
  | 'ALREADY_EXISTS'
  | 'CORRUPT'
  | 'UNSUPPORTED'
  | 'LOCKED'
  | 'EMBEDDER_UNAVAILABLE'
  | 'PATH_UNSAFE'
  | 'READ_ONLY'
  | 'VERSION';

export class MemError extends Error {
  readonly code: MemErrorCode;
  readonly details?: Record<string, unknown>;
  constructor(code: MemErrorCode, message: string, details?: Record<string, unknown>) {
    super(message);
    this.name = 'MemError';
    this.code = code;
    this.details = details;
  }
  toJSON() {
    return { error: this.code, message: this.message, details: this.details };
  }
}

export function invalid(message: string, details?: Record<string, unknown>): never {
  throw new MemError('INVALID_ARGUMENT', message, details);
}
