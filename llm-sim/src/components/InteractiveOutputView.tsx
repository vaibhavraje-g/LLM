import React, { useState, useMemo } from 'react';
import { BarChart, Activity, Settings2, Info } from 'lucide-react';
import { softmax } from '../utils/math';

interface InteractiveOutputViewProps {
  logits: number[];
  vocab: string[];
  initialTemperature?: number;
  initialTopK?: number;
  initialTopP?: number;
}

export const InteractiveOutputView: React.FC<InteractiveOutputViewProps> = ({
  logits,
  vocab,
  initialTemperature = 1.0,
  initialTopK = 5,
  initialTopP = 0.9,
}) => {
  // Local state for "what-if" analysis
  const [temperature, setTemperature] = useState(initialTemperature);
  const [topK, setTopK] = useState(initialTopK);
  const [topP, setTopP] = useState(initialTopP);

  // Calculate probabilities based on current logits and temperature
  const probabilities = useMemo(() => {
    return softmax(logits, temperature);
  }, [logits, temperature]);

  // Combine with vocab and sort
  const sortedTokens = useMemo(() => {
    return probabilities
      .map((prob, i) => ({ text: vocab[i], prob, id: i }))
      .sort((a, b) => b.prob - a.prob);
  }, [probabilities, vocab]);

  // Apply Sampling Logic (Top-K & Top-P) to determine which tokens are "active"
  const processedTokens = useMemo(() => {
    let cumulativeProb = 0;
    return sortedTokens.map((token, index) => {
      cumulativeProb += token.prob;
      
      // Top-K Logic
      const isWithinTopK = index < topK;
      
      // Top-P Logic (Nucleus)
      // We include the token that pushes us over the threshold
      const isWithinTopP = cumulativeProb - token.prob < topP;

      const isActive = isWithinTopK && isWithinTopP;

      return { ...token, isActive, cumulativeProb };
    });
  }, [sortedTokens, topK, topP]);

  // Filter for display (show top 10 or so, plus a "others" bar if needed)
  const displayTokens = processedTokens.slice(0, 10);

  return (
    <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-200 mb-6 animate-in fade-in duration-500">
      <div className="flex items-center gap-2 mb-4 border-b pb-2">
        <Activity className="text-blue-600" size={18} />
        <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wide">Interactive Sampling</h3>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls */}
        <div className="space-y-6 bg-slate-50 p-4 rounded-lg border border-slate-100">
          <div className="flex items-center gap-2 text-xs font-bold text-slate-500 uppercase mb-2">
            <Settings2 size={14} /> Parameters
          </div>

          {/* Temperature */}
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="font-medium text-slate-700">Temperature</span>
              <span className="font-mono bg-white px-1.5 rounded border border-slate-200">{temperature.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="2.0"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full accent-blue-600 h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer"
            />
            <p className="text-[10px] text-slate-400 mt-1">
              Controls randomness. Low = confident, High = creative.
            </p>
          </div>

          {/* Top-K */}
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="font-medium text-slate-700">Top-K</span>
              <span className="font-mono bg-white px-1.5 rounded border border-slate-200">{topK}</span>
            </div>
            <input
              type="range"
              min="1"
              max={vocab.length}
              step="1"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value))}
              className="w-full accent-green-600 h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer"
            />
            <p className="text-[10px] text-slate-400 mt-1">
              Limits to the top K most likely tokens.
            </p>
          </div>

          {/* Top-P */}
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="font-medium text-slate-700">Top-P (Nucleus)</span>
              <span className="font-mono bg-white px-1.5 rounded border border-slate-200">{topP.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={topP}
              onChange={(e) => setTopP(parseFloat(e.target.value))}
              className="w-full accent-purple-600 h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer"
            />
            <p className="text-[10px] text-slate-400 mt-1">
              Limits to the smallest set of tokens summing to probability P.
            </p>
          </div>
        </div>

        {/* Visualization */}
        <div className="lg:col-span-2 flex flex-col h-full">
          <div className="flex items-center justify-between mb-2">
             <div className="text-xs font-bold text-slate-500 uppercase flex items-center gap-1">
                <BarChart size={14} /> Probability Distribution
             </div>
             <div className="flex gap-2 text-[10px]">
                <span className="flex items-center gap-1"><div className="w-2 h-2 bg-blue-500 rounded-full"></div> Selected Pool</span>
                <span className="flex items-center gap-1"><div className="w-2 h-2 bg-slate-300 rounded-full"></div> Filtered Out</span>
             </div>
          </div>

          <div className="flex-1 bg-white border border-slate-100 rounded-lg p-4 relative min-h-[200px] flex items-end gap-2 overflow-x-auto">
             {displayTokens.map((token, i) => (
               <div key={i} className="flex-1 flex flex-col items-center group min-w-[40px]">
                  {/* Tooltip */}
                  <div className="opacity-0 group-hover:opacity-100 absolute bottom-full mb-2 bg-slate-800 text-white text-[10px] p-1.5 rounded pointer-events-none transition-opacity z-10 whitespace-nowrap">
                     "{token.text}": {(token.prob * 100).toFixed(2)}%
                     {!token.isActive && " (Filtered)"}
                  </div>

                  {/* Bar */}
                  <div className="w-full relative flex items-end justify-center h-[180px] bg-slate-50 rounded-t-sm overflow-hidden">
                     <div 
                        className={`w-full transition-all duration-500 ${token.isActive ? 'bg-blue-500 hover:bg-blue-600' : 'bg-slate-300 opacity-50'}`}
                        style={{ height: `${token.prob * 100}%` }}
                     />
                  </div>
                  
                  {/* Label */}
                  <div className={`mt-2 text-xs font-mono font-bold truncate w-full text-center ${token.isActive ? 'text-slate-700' : 'text-slate-400'}`}>
                    {token.text}
                  </div>
                  <div className="text-[9px] text-slate-400">
                    {(token.prob * 100).toFixed(0)}%
                  </div>
               </div>
             ))}
             
             {/* "Others" Bar */}
             {processedTokens.length > 10 && (
                <div className="flex-1 flex flex-col items-center min-w-[40px] opacity-50">
                   <div className="w-full h-[180px] bg-slate-50 rounded-t-sm flex items-end">
                      <div className="w-full bg-slate-300 h-4"></div>
                   </div>
                   <div className="mt-2 text-xs font-mono text-slate-400">...</div>
                </div>
             )}
          </div>
          
          <div className="mt-2 p-2 bg-blue-50 rounded border border-blue-100 text-xs text-blue-800 flex items-start gap-2">
             <Info size={14} className="mt-0.5 flex-shrink-0" />
             <p>
               <strong>Insight:</strong> 
               {temperature > 1.2 ? " High temperature makes the distribution flatter, giving rare words a chance." : 
                temperature < 0.5 ? " Low temperature makes the model very confident in the top choice." : 
                " Balanced temperature allows for some creativity while staying coherent."}
               {topK < 5 && " Top-K is very restrictive."}
             </p>
          </div>
        </div>
      </div>
    </div>
  );
};
