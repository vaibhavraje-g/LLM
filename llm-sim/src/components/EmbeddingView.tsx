import React, { useState } from 'react';
import type { Token, Matrix } from '../types';

interface EmbeddingViewProps {
  tokens: Token[];
  embeddings: Matrix;
  positionalEncodings: Matrix;
  finalInput: Matrix;
}

export const EmbeddingView: React.FC<EmbeddingViewProps> = ({ 
  tokens, 
  embeddings, 
  positionalEncodings, 
  finalInput 
}) => {
  const [showMath, setShowMath] = useState(false);

  return (
    <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 mb-6">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-bold text-slate-800">2. Embeddings & Positional Encoding</h3>
        <button 
          onClick={() => setShowMath(!showMath)}
          className="text-xs px-3 py-1 bg-slate-100 hover:bg-slate-200 rounded-full text-slate-600 font-medium transition-colors"
        >
          {showMath ? "Hide Math" : "Show Math"}
        </button>
      </div>
      
      <p className="text-sm text-slate-600 mb-4">
        Token IDs are converted into dense vectors (Embeddings). Positional encodings are added so the model knows the order of tokens.
      </p>

      <div className="overflow-x-auto">
        <table className="w-full text-sm text-left">
          <thead>
            <tr className="border-b border-slate-200">
              <th className="py-2 px-3 font-medium text-slate-500">Token</th>
              <th className="py-2 px-3 font-medium text-slate-500">Token Embedding (E)</th>
              <th className="py-2 px-3 font-medium text-slate-500 text-center">+</th>
              <th className="py-2 px-3 font-medium text-slate-500">Positional Encoding (P)</th>
              <th className="py-2 px-3 font-medium text-slate-500 text-center">=</th>
              <th className="py-2 px-3 font-medium text-slate-500">Final Input (X)</th>
            </tr>
          </thead>
          <tbody>
            {tokens.map((token, i) => (
              <tr key={i} className="border-b border-slate-100 hover:bg-slate-50">
                <td className="py-3 px-3">
                  <div className={`inline-block px-2 py-1 rounded text-xs font-bold ${token.color}`}>
                    {token.text}
                  </div>
                </td>
                <td className="py-3 px-3 font-mono text-xs text-blue-600">
                  [{embeddings[i]?.join(", ")}]
                </td>
                <td className="py-3 px-3 text-center text-slate-400">+</td>
                <td className="py-3 px-3 font-mono text-xs text-purple-600">
                  [{positionalEncodings[i]?.join(", ")}]
                </td>
                <td className="py-3 px-3 text-center text-slate-400">=</td>
                <td className="py-3 px-3 font-mono text-xs font-bold text-slate-800">
                  [{finalInput[i]?.join(", ")}]
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {showMath && (
        <div className="mt-4 p-4 bg-slate-50 rounded-lg border border-slate-200 text-xs font-mono">
          <p className="mb-2 font-bold text-slate-700">Math Details:</p>
          <p>X[i] = Embedding(token_id[i]) + PositionalEncoding(i)</p>
          <p className="mt-2 text-slate-500">
            Positional encodings use sine/cosine functions to create unique vectors for each position.
          </p>
        </div>
      )}
    </div>
  );
};
