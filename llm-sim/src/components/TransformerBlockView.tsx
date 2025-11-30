import React, { useState } from 'react';
import { MathExplanation } from './MathExplanation';

// Define types locally if not importing to avoid circular deps or just for simplicity in this view
type Matrix = number[][];

interface TransformerBlockViewProps {
  data: {
    Q: Matrix;
    K: Matrix;
    V: Matrix;
    attentionScores: Matrix;
    attentionWeights: Matrix;
    ffnHidden: Matrix;
    finalBlockOutput: Matrix;
  } | null;
  dim: number;
}

export const TransformerBlockView: React.FC<TransformerBlockViewProps> = ({ data, dim }) => {
  const [activeTab, setActiveTab] = useState<'attention' | 'ffn'>('attention');
  const [showMath, setShowMath] = useState(false);

  if (!data) return null;

  // We visualize the last step (most recent token processing) for simplicity in the block view
  // or we could show the whole sequence. Let's show the last token's perspective (last row of matrices)
  
  const lastTokenIndex = data.attentionWeights.length - 1;
  const currentScores = data.attentionScores[lastTokenIndex] || [];
  const currentWeights = data.attentionWeights[lastTokenIndex] || [];
  
  // For Q, K, V, these are arrays of vectors (one per token). 
  // Let's show the vectors for the current token being generated/processed.
  const currentQ = data.Q[lastTokenIndex] || [];
  const currentK = data.K[lastTokenIndex] || []; // Actually K is usually all previous tokens. 
  const currentV = data.V[lastTokenIndex] || [];

  return (
    <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-200 mb-6">
      <div className="flex justify-between items-center mb-4 border-b pb-2">
        <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wide">3. Transformer Internals</h3>
        <div className="flex gap-2">
          <button 
            onClick={() => setActiveTab('attention')}
            className={`px-3 py-1 rounded-md text-xs font-bold transition-colors ${activeTab === 'attention' ? 'bg-blue-100 text-blue-700' : 'text-slate-500 hover:bg-slate-100'}`}
          >
            Self-Attention
          </button>
          <button 
            onClick={() => setActiveTab('ffn')}
            className={`px-3 py-1 rounded-md text-xs font-bold transition-colors ${activeTab === 'ffn' ? 'bg-purple-100 text-purple-700' : 'text-slate-500 hover:bg-slate-100'}`}
          >
            Feed-Forward
          </button>
        </div>
      </div>

      {activeTab === 'attention' && (
        <div className="space-y-6 animate-in fade-in duration-300">
          <div className="bg-blue-50 p-3 rounded border border-blue-100 text-xs text-blue-800 leading-relaxed">
            <strong>Concept:</strong> The model compares the current token (Query) against all previous tokens (Keys) to determine relevance. It then aggregates information (Values) based on these relevance scores.
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <VectorDisplay title="Query Vector (Current)" vector={currentQ} color="text-blue-600" />
            {/* For K and V, we ideally show a matrix, but let's show the current token's K/V for symmetry, or maybe the last few */}
            <VectorDisplay title="Key Vector (Current)" vector={currentK} color="text-emerald-600" />
            <VectorDisplay title="Value Vector (Current)" vector={currentV} color="text-purple-600" />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="text-[10px] font-bold uppercase text-slate-400 mb-2">Attention Scores (Dot Product)</h4>
              <div className="flex gap-1 overflow-x-auto pb-2">
                {currentScores.map((val, i) => (
                   <div key={i} className="flex flex-col items-center min-w-[2rem]">
                      <div className={`w-full h-8 rounded flex items-center justify-center text-[10px] font-mono border ${val === -1e9 ? 'bg-gray-100 text-gray-300' : 'bg-yellow-50 border-yellow-200 text-yellow-800'}`}>
                        {val === -1e9 ? '-∞' : val.toFixed(1)}
                      </div>
                      <span className="text-[9px] text-gray-400 mt-1">T{i}</span>
                   </div>
                ))}
              </div>
            </div>

            <div>
              <h4 className="text-[10px] font-bold uppercase text-slate-400 mb-2">Attention Weights (Softmax)</h4>
              <div className="flex gap-1 overflow-x-auto pb-2">
                {currentWeights.map((val, i) => (
                   <div key={i} className="flex flex-col items-center min-w-[2rem]">
                      <div className="w-full h-8 rounded border border-blue-100 flex items-center justify-center text-[10px] font-mono relative overflow-hidden">
                        <div className="absolute bottom-0 left-0 right-0 bg-blue-500 opacity-20" style={{ height: `${val * 100}%` }}></div>
                        <span className="relative z-10">{val.toFixed(2)}</span>
                      </div>
                      <span className="text-[9px] text-gray-400 mt-1">T{i}</span>
                   </div>
                ))}
              </div>
            </div>
          </div>
          
          {showMath && (
             <MathExplanation 
                title="Scaled Dot-Product Attention"
                formula="Attention(Q, K, V) = softmax(QK^T / √d_k)V"
                variables={{
                  "Q": "Query Matrix (What I'm looking for)",
                  "K": "Key Matrix (What I match against)",
                  "V": "Value Matrix (What I retrieve)",
                  "d_k": `Dimension size (${dim})`
                }}
             />
          )}
        </div>
      )}

      {activeTab === 'ffn' && (
        <div className="space-y-6 animate-in fade-in duration-300">
          <div className="bg-purple-50 p-3 rounded border border-purple-100 text-xs text-purple-800 leading-relaxed">
            <strong>Concept:</strong> The Feed-Forward Network (FFN) is where the model "thinks" about the information it just gathered. It projects the data into a higher dimension (usually 4x) and back down, applying non-linearity (ReLU) to learn complex patterns.
          </div>
          
          <div className="grid grid-cols-1 gap-4">
             <div className="space-y-2">
                <h4 className="text-[10px] font-bold uppercase text-slate-400">Hidden Layer Activations (ReLU)</h4>
                <div className="flex flex-wrap gap-1">
                  {data.ffnHidden[lastTokenIndex]?.map((val, i) => (
                    <div key={i} className={`w-8 h-8 rounded border flex items-center justify-center text-[9px] font-mono ${val > 0 ? 'bg-green-50 border-green-200 text-green-700 font-bold' : 'bg-gray-50 text-gray-300'}`}>
                      {val.toFixed(1)}
                    </div>
                  ))}
                </div>
                <p className="text-[10px] text-slate-400 italic">Values &le; 0 are deactivated (ReLU).</p>
             </div>
          </div>

          {showMath && (
             <MathExplanation 
                title="Position-wise Feed-Forward"
                formula="FFN(x) = max(0, xW_1 + b_1)W_2 + b_2"
                variables={{
                  "x": "Input vector from Attention",
                  "W_1, W_2": "Learned Weight Matrices",
                  "max(0, ...)": "ReLU Activation Function"
                }}
             />
          )}
        </div>
      )}

      <div className="mt-4 pt-2 border-t border-slate-100 flex justify-end">
        <button 
          onClick={() => setShowMath(!showMath)}
          className="text-[10px] font-bold text-slate-400 hover:text-blue-600 uppercase tracking-wider flex items-center gap-1 transition-colors"
        >
          <Calculator size={12} />
          {showMath ? "Hide Math" : "Show Math"}
        </button>
      </div>
    </div>
  );
};

const VectorDisplay = ({ title, vector, color }: { title: string, vector: number[], color: string }) => (
  <div className="bg-slate-50 p-2 rounded border border-slate-100">
    <h4 className="text-[10px] font-bold uppercase text-slate-400 mb-1">{title}</h4>
    <div className={`font-mono text-[10px] ${color} break-all`}>
      [{vector.map(v => v.toFixed(2)).join(", ")}]
    </div>
  </div>
);

// Icon helper
const Calculator = ({ size }: { size: number }) => (
  <svg 
    xmlns="http://www.w3.org/2000/svg" 
    width={size} 
    height={size} 
    viewBox="0 0 24 24" 
    fill="none" 
    stroke="currentColor" 
    strokeWidth="2" 
    strokeLinecap="round" 
    strokeLinejoin="round"
  >
    <rect width="16" height="20" x="4" y="2" rx="2"/>
    <line x1="8" x2="16" y1="6" y2="6"/>
    <line x1="16" x2="16" y1="14" y2="18"/>
    <path d="M16 10h.01"/>
    <path d="M12 10h.01"/>
    <path d="M8 10h.01"/>
    <path d="M12 14h.01"/>
    <path d="M8 14h.01"/>
    <path d="M12 18h.01"/>
    <path d="M8 18h.01"/>
  </svg>
);

