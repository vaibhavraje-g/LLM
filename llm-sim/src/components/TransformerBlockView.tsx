import React, { useState } from 'react';
import { TransformerBlockState, Matrix } from '../types';

interface TransformerBlockViewProps {
  state: TransformerBlockState | null;
}

export const TransformerBlockView: React.FC<TransformerBlockViewProps> = ({ state }) => {
  const [activeTab, setActiveTab] = useState<'attention' | 'ffn'>('attention');
  const [showMath, setShowMath] = useState(false);

  if (!state) return null;

  const head = state.attentionHeads[0]; // Visualizing first head for simplicity

  return (
    <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 mb-6">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-bold text-slate-800">3. Transformer Block</h3>
        <div className="flex gap-2">
          <button 
            onClick={() => setActiveTab('attention')}
            className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${activeTab === 'attention' ? 'bg-blue-100 text-blue-700' : 'text-slate-600 hover:bg-slate-100'}`}
          >
            Self-Attention
          </button>
          <button 
            onClick={() => setActiveTab('ffn')}
            className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${activeTab === 'ffn' ? 'bg-blue-100 text-blue-700' : 'text-slate-600 hover:bg-slate-100'}`}
          >
            Feed-Forward
          </button>
        </div>
      </div>

      {activeTab === 'attention' && (
        <div className="space-y-6">
          <p className="text-sm text-slate-600">
            Self-attention allows tokens to look at each other. We compute Query (Q), Key (K), and Value (V) vectors, then calculate attention scores to decide how much focus to put on other tokens.
          </p>

          <div className="grid grid-cols-3 gap-4">
            <MatrixDisplay title="Query (Q)" matrix={head.Q} color="text-blue-600" />
            <MatrixDisplay title="Key (K)" matrix={head.K} color="text-emerald-600" />
            <MatrixDisplay title="Value (V)" matrix={head.V} color="text-purple-600" />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="text-xs font-semibold uppercase text-slate-500 mb-2">Attention Scores (Q · K^T)</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-xs text-center border-collapse">
                  <tbody>
                    {head.scores.map((row, i) => (
                      <tr key={i}>
                        {row.map((val, j) => (
                          <td key={j} className={`p-2 border border-slate-100 ${val === -1e9 ? 'text-slate-300 bg-slate-50' : 'bg-yellow-50'}`}>
                            {val === -1e9 ? '-∞' : val.toFixed(2)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div>
              <h4 className="text-xs font-semibold uppercase text-slate-500 mb-2">Attention Weights (Softmax)</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-xs text-center border-collapse">
                  <tbody>
                    {head.weights.map((row, i) => (
                      <tr key={i}>
                        {row.map((val, j) => (
                          <td key={j} className="p-2 border border-slate-100" style={{ backgroundColor: `rgba(59, 130, 246, ${val})` }}>
                            {val.toFixed(2)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="bg-slate-50 p-4 rounded-lg">
             <h4 className="text-xs font-semibold uppercase text-slate-500 mb-2">Weighted Sum (New Representation)</h4>
             <MatrixDisplay title="" matrix={head.weightedSum} color="text-slate-800" />
          </div>
        </div>
      )}

      {activeTab === 'ffn' && (
        <div className="space-y-6">
          <p className="text-sm text-slate-600">
            The Feed-Forward Network (FFN) processes each token independently to extract more complex features. It's usually a simple 2-layer neural network with ReLU activation.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
             <MatrixDisplay title="FFN Input (from Attention)" matrix={state.ffnInput} color="text-slate-600" />
             <MatrixDisplay title="FFN Output" matrix={state.ffnOutput} color="text-slate-800 font-bold" />
          </div>
        </div>
      )}

      <div className="mt-4 pt-4 border-t border-slate-100">
        <button 
          onClick={() => setShowMath(!showMath)}
          className="text-xs text-slate-500 hover:text-blue-600 underline"
        >
          {showMath ? "Hide Math Details" : "Show Math Details"}
        </button>
        
        {showMath && (
          <div className="mt-2 p-3 bg-slate-50 rounded text-xs font-mono text-slate-600 space-y-1">
            <p>Attention(Q, K, V) = softmax(QK^T / √d_k)V</p>
            <p>FFN(x) = max(0, xW1 + b1)W2 + b2</p>
          </div>
        )}
      </div>
    </div>
  );
};

const MatrixDisplay = ({ title, matrix, color }: { title: string, matrix: Matrix, color: string }) => (
  <div>
    {title && <h4 className="text-xs font-semibold uppercase text-slate-500 mb-2">{title}</h4>}
    <div className={`font-mono text-xs ${color} bg-slate-50 p-2 rounded border border-slate-100 overflow-x-auto`}>
      {matrix.map((row, i) => (
        <div key={i} className="whitespace-nowrap">
          [{row.map(v => v.toFixed(2)).join(", ")}]
        </div>
      ))}
    </div>
  </div>
);
