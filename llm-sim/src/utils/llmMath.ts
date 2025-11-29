import { Matrix, EmbeddingVector, AttentionHead, TransformerBlockState } from '../types';

// Helper to generate random matrix with fixed seed (simulated by simple hash)
export const generateRandomMatrix = (rows: number, cols: number, seed: number = 1): Matrix => {
  const matrix: Matrix = [];
  for (let i = 0; i < rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < cols; j++) {
      // Simple pseudo-random generator
      const x = Math.sin(seed * 1000 + i * 100 + j) * 10000;
      const val = parseFloat((x - Math.floor(x)).toFixed(2)); // 0.00 to 0.99
      row.push(val * 2 - 1); // -1 to 1
    }
    matrix.push(row);
  }
  return matrix;
};

export const getEmbeddings = (tokenIds: number[], embeddingMatrix: Matrix): Matrix => {
  return tokenIds.map(id => embeddingMatrix[id] || new Array(embeddingMatrix[0].length).fill(0));
};

export const addPositionalEncoding = (embeddings: Matrix): { finalInput: Matrix, positionalEncodings: Matrix } => {
  const dim = embeddings[0].length;
  const positionalEncodings: Matrix = embeddings.map((_, pos) => {
    return Array.from({ length: dim }, (_, i) => {
      // Simple sinusoidal encoding simulation
      const val = Math.sin(pos / Math.pow(10000, (2 * i) / dim));
      return parseFloat(val.toFixed(2));
    });
  });

  const finalInput = embeddings.map((row, i) => 
    row.map((val, j) => parseFloat((val + positionalEncodings[i][j]).toFixed(2)))
  );

  return { finalInput, positionalEncodings };
};

export const dotProduct = (a: number[], b: number[]): number => {
  return a.reduce((sum, val, i) => sum + val * b[i], 0);
};

export const matMul = (a: Matrix, b: Matrix): Matrix => {
  const rowsA = a.length;
  const colsA = a[0].length;
  const rowsB = b.length;
  const colsB = b[0].length;

  if (colsA !== rowsB) throw new Error("Matrix dimension mismatch");

  const result: Matrix = [];
  for (let i = 0; i < rowsA; i++) {
    const row: number[] = [];
    for (let j = 0; j < colsB; j++) {
      let sum = 0;
      for (let k = 0; k < colsA; k++) {
        sum += a[i][k] * b[k][j];
      }
      row.push(parseFloat(sum.toFixed(2)));
    }
    result.push(row);
  }
  return result;
};

export const softmax = (logits: number[]): number[] => {
  const max = Math.max(...logits);
  const exps = logits.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => parseFloat((v / sum).toFixed(4)));
};

export const computeAttention = (input: Matrix, W_Q: Matrix, W_K: Matrix, W_V: Matrix): AttentionHead => {
  const Q = matMul(input, W_Q);
  const K = matMul(input, W_K);
  const V = matMul(input, W_V);
  
  const d_k = Q[0].length;
  const scores: Matrix = [];
  const weights: Matrix = [];
  const weightedSum: Matrix = [];

  // Compute scores and weights
  for (let i = 0; i < Q.length; i++) {
    const scoreRow: number[] = [];
    for (let j = 0; j < K.length; j++) {
      // Masking: only look at past tokens (j <= i)
      if (j <= i) {
        const rawScore = dotProduct(Q[i], K[j]) / Math.sqrt(d_k);
        scoreRow.push(parseFloat(rawScore.toFixed(2)));
      } else {
        scoreRow.push(-1e9); // Mask
      }
    }
    scores.push(scoreRow);
    weights.push(softmax(scoreRow));
  }

  // Compute weighted sum
  for (let i = 0; i < weights.length; i++) {
    const row: number[] = new Array(V[0].length).fill(0);
    for (let j = 0; j < weights[i].length; j++) {
      const weight = weights[i][j];
      for (let k = 0; k < V[j].length; k++) {
        row[k] += weight * V[j][k];
      }
    }
    weightedSum.push(row.map(v => parseFloat(v.toFixed(2))));
  }

  return { Q, K, V, scores, weights, weightedSum };
};

export const computeFFN = (input: Matrix, W1: Matrix, b1: number[], W2: Matrix, b2: number[]): Matrix => {
  // Layer 1: xW1 + b1
  const hidden = matMul(input, W1).map((row) => 
    row.map((val, i) => Math.max(0, val + b1[i])) // ReLU
  );

  // Layer 2: hW2 + b2
  const output = matMul(hidden, W2).map((row) => 
    row.map((val, i) => parseFloat((val + b2[i]).toFixed(2)))
  );

  return output;
};
