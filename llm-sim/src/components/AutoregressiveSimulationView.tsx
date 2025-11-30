import React from 'react';
import type { Token } from '../types';

interface AutoregressiveSimulationViewProps {
  generatedTokens: Token[];
  maxTokens: number;
}

export const AutoregressiveSimulationView: React.FC<AutoregressiveSimulationViewProps> = ({ generatedTokens, maxTokens }) => {
  return (
    <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 mb-6">
      <h3 className="text-lg font-bold text-slate-800 mb-4">5. Autoregressive Generation Loop</h3>
      <p className="text-sm text-slate-600 mb-4">
        The model generates one token at a time. Each new token is appended to the input, and the whole process repeats.
      </p>

      <div className="relative pt-8 pb-4">
        <div className="absolute top-1/2 left-0 w-full h-1 bg-slate-100 -translate-y-1/2 rounded-full" />
        
        <div className="flex gap-4 overflow-x-auto pb-4 relative z-10 px-2">
          {/* Start Node */}
          <div className="flex flex-col items-center min-w-[80px]">
            <div className="w-4 h-4 rounded-full bg-blue-500 mb-2 ring-4 ring-white" />
            <span className="text-xs font-bold text-slate-500">Start</span>
          </div>

          {generatedTokens.map((token, i) => (
            <div key={i} className="flex flex-col items-center min-w-[80px] animate-fade-in-up">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold mb-2 ring-4 ring-white shadow-sm ${token.color}`}>
                {i + 1}
              </div>
              <span className="text-xs font-bold text-slate-700 px-2 py-1 bg-white rounded border border-slate-200 shadow-sm">
                {token.text}
              </span>
              <span className="text-[10px] text-slate-400 mt-1">Step {i + 1}</span>
            </div>
          ))}

          {/* Future Nodes Placeholder */}
          {Array.from({ length: Math.max(0, maxTokens - generatedTokens.length) }).map((_, i) => (
            <div key={`future-${i}`} className="flex flex-col items-center min-w-[80px] opacity-30">
              <div className="w-4 h-4 rounded-full bg-slate-300 mb-2 ring-4 ring-white" />
              <span className="text-xs text-slate-400">...</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
