import React from 'react';
import { Calculator } from 'lucide-react';

interface MathExplanationProps {
  formula: string;
  description?: string;
  variables?: Record<string, string>;
  title?: string;
}

export const MathExplanation: React.FC<MathExplanationProps> = ({ 
  formula, 
  description, 
  variables,
  title = "Mathematical Formula"
}) => {
  return (
    <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 my-2">
      <div className="flex items-center gap-2 mb-2 text-slate-500 text-xs font-bold uppercase tracking-wider">
        <Calculator size={12} />
        {title}
      </div>
      
      <div className="bg-white p-3 rounded border border-slate-200 text-center font-mono text-sm text-slate-800 mb-3 shadow-sm overflow-x-auto">
        {formula}
      </div>

      {description && (
        <p className="text-xs text-slate-600 mb-3 leading-relaxed">
          {description}
        </p>
      )}

      {variables && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
          {Object.entries(variables).map(([symbol, desc]) => (
            <div key={symbol} className="flex items-start gap-2">
              <span className="font-mono font-bold text-blue-600 bg-blue-50 px-1.5 rounded">{symbol}</span>
              <span className="text-slate-600">{desc}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
