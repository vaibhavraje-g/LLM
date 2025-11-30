export const generateMatrix = (rows: number, cols: number, seed: number = 1): number[][] => {
  const matrix: number[][] = [];
  for (let i = 0; i < rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < cols; j++) {
      const val = Math.sin(i * 10 + j * 20 + seed) * 0.5;
      row.push(val);
    }
    matrix.push(row);
  }
  return matrix;
};

export const addVectors = (v1: number[], v2: number[]): number[] => {
  return v1.map((val, i) => val + (v2[i] || 0));
};

export const matMulVector = (vec: number[], matrix: number[][]): number[] => {
  const result: number[] = [];
  const cols = matrix[0].length;
  for (let j = 0; j < cols; j++) {
    let sum = 0;
    for (let i = 0; i < vec.length; i++) {
      sum += vec[i] * matrix[i][j];
    }
    result.push(sum);
  }
  return result;
};

export const dotProduct = (v1: number[], v2: number[]): number => {
  return v1.reduce((sum, val, i) => sum + val * v2[i], 0);
};

export const softmax = (logits: number[], temperature: number = 1.0): number[] => {
  const maxLogit = Math.max(...logits); // Numerical stability
  const exps = logits.map(l => Math.exp((l - maxLogit) / temperature));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sumExps);
};

export const relu = (vec: number[]): number[] => vec.map(x => Math.max(0, x));
