export type Token = {
  id: number;
  text: string;
  color?: string;
};

export type EmbeddingVector = number[];

export type Matrix = number[][];

export type AttentionHead = {
  Q: Matrix;
  K: Matrix;
  V: Matrix;
  scores: Matrix;
  weights: Matrix;
  weightedSum: Matrix;
};

export type TransformerBlockState = {
  inputEmbeddings: Matrix;
  positionalEncodings: Matrix;
  finalInput: Matrix;
  attentionHeads: AttentionHead[];
  ffnInput: Matrix;
  ffnOutput: Matrix;
  finalOutput: Matrix;
};

export type SimulationStep = 
  | 'IDLE' 
  | 'TOKENIZING' 
  | 'EMBEDDING' 
  | 'ATTENTION' 
  | 'FFN' 
  | 'OUTPUT' 
  | 'APPEND';

export type SimulationState = {
  step: SimulationStep;
  input: string;
  tokens: Token[];
  generatedTokens: Token[];
  transformerState: TransformerBlockState | null;
  logits: number[];
  probabilities: { token: Token; prob: number }[];
  selectedToken: Token | null;
};

export type ModelConfig = {
  vocabSize: number;
  embeddingDim: number;
  numHeads: number;
  maxTokens: number;
  temperature: number;
};
