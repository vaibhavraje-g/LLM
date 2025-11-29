import { Token } from '../types';

export const VOCABULARY: string[] = [
  "<PAD>", "<START>", "<END>", "<UNK>", 
  "I", "like", "cats", "dogs", "and", "you", "are", "cute", "very", "much", ".", "!"
];

export const TOKEN_COLORS: Record<string, string> = {
  "<PAD>": "bg-gray-200 text-gray-500",
  "<START>": "bg-green-100 text-green-700",
  "<END>": "bg-red-100 text-red-700",
  "<UNK>": "bg-yellow-100 text-yellow-700",
  "default": "bg-blue-50 text-blue-700"
};

export const DEFAULT_CONFIG = {
  vocabSize: VOCABULARY.length,
  embeddingDim: 4,
  numHeads: 1,
  maxTokens: 10,
  temperature: 1.0,
};

export const TOKENS: Token[] = VOCABULARY.map((text, id) => ({
  id,
  text,
  color: TOKEN_COLORS[text] || TOKEN_COLORS["default"]
}));
