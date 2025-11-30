import React from 'react';
import type { Token } from '../types';

interface TokenizationViewProps {
  input: string;
  tokens: Token[];
}

export const TokenizationView: React.FC<TokenizationViewProps> = ({ input, tokens }) => {
  return (
    <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 mb-6">
      <h3 className="text-lg font-bold text-slate-800 mb-4">1. Tokenization</h3>
      <p className="text-sm text-slate-600 mb-4">
        The raw text is split into smaller units called tokens. Each token is assigned a unique numeric ID from the vocabulary.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h4 className="text-xs font-semibold uppercase text-slate-500 mb-2">Raw Input</h4>
          <div className="p-3 bg-slate-100 rounded-lg font-mono text-sm min-h-[50px] flex items-center">
            "{input}"
          </div>
        </div>

        <div>
          <h4 className="text-xs font-semibold uppercase text-slate-500 mb-2">Token Sequence</h4>
          <div className="flex flex-wrap gap-2">
            {tokens.map((token, idx) => (
              <div key={idx} className={`flex flex-col items-center p-2 rounded-lg border ${token.color} border-opacity-20`}>
                <span className="font-bold text-xs">{token.text}</span>
                <span className="text-[10px] opacity-70">ID: {token.id}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
