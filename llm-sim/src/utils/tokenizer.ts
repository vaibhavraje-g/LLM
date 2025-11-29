import { Token } from '../types';
import { TOKENS, VOCABULARY } from './constants';

export const tokenize = (text: string): Token[] => {
  // Simple whitespace and punctuation splitting
  // This is a "toy" tokenizer for educational purposes
  const rawWords = text
    .replace(/([.,!])/g, " $1 ") // Add spaces around punctuation
    .trim()
    .split(/\s+/);

  const tokens: Token[] = [];

  // Always start with <START>
  tokens.push(TOKENS.find(t => t.text === "<START>")!);

  for (const word of rawWords) {
    if (!word) continue;
    
    const found = TOKENS.find(t => t.text === word);
    if (found) {
      tokens.push(found);
    } else {
      tokens.push(TOKENS.find(t => t.text === "<UNK>")!);
    }
  }

  return tokens;
};

export const detokenize = (tokens: Token[]): string => {
  return tokens
    .filter(t => t.text !== "<START>" && t.text !== "<PAD>")
    .map(t => t.text)
    .join(" ")
    .replace(/\s+([.,!])/g, "$1"); // Remove space before punctuation
};
