import React from 'react';
import { Token } from '../types';

interface OutputLayerViewProps {
  logits: number[];
  probabilities: { token: Token; prob: number }[];
  selectedToken: Token | null;
}

export const OutputLayerView: React.FC<OutputLayerViewProps> = ({ logits, probabilities, selectedToken }) => {
  // Sort probabilities for display, but keep top 5
  const topProbs = [...probabilities].sort((a, b) => b.prob - a.prob).slice(0, 5);

  return (
    <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 mb-6">
      <h3 className="text-lg font-bold text-slate-800 mb-4">4. Output Layer & Sampling</h3>
      <p className="text-sm text-slate-600 mb-4">
        The model predicts the next token by calculating a score (logit) for every word in the vocabulary, converting them to probabilities, and then picking one.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div>
          <h4 className="text-xs font-semibold uppercase text-slate-500 mb-2">Top 5 Predictions</h4>
          <div className="space-y-2">
            {topProbs.map((p, i) => (
              <div key={i} className="flex items-center gap-3">
                <div className="w-16 text-right font-mono text-xs text-slate-500">{(p.prob * 100).toFixed(1)}%</div>
                <div className="flex-1 h-6 bg-slate-100 rounded-full overflow-hidden relative">
                  <div 
                    className="h-full bg-blue-500 absolute top-0 left-0 transition-all duration-500"
                    style={{ width: `${p.prob * 100}%` }}
                  />
                  <div className="absolute inset-0 flex items-center pl-2 text-xs font-bold z-10 text-slate-700">
                    {p.token.text}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="flex flex-col items-center justify-center p-4 bg-slate-50 rounded-xl border border-slate-200">
          <h4 className="text-xs font-semibold uppercase text-slate-500 mb-3">Selected Next Token</h4>
          {selectedToken ? (
            <div className={`text-2xl font-bold px-6 py-3 rounded-lg shadow-sm ${selectedToken.color} animate-bounce-short`}>
              {selectedToken.text}
            </div>
          ) : (
            <div className="text-slate-400 italic">Waiting for generation...</div>
          )}
          <p className="text-xs text-slate-400 mt-2 text-center max-w-[200px]">
            Based on temperature and random sampling (or argmax).
          </p>
        </div>
      </div>
    </div>
  );
};
