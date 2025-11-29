import React, { useState, useEffect, useRef, useMemo } from 'react';
import { 
  Play, SkipForward, RefreshCw, Settings, ChevronDown, ChevronRight, 
  Info, Database, ArrowRight, CornerDownLeft, X, 
  Brain, Calculator,  List, Type, Search
} from 'lucide-react';

// --- Constants & Config ---

const VOCAB = [
  "<START>", "I", "like", "love", "hate", "cats", "dogs", "code", "math", 
  "pizza", "is", "are", "fun", "hard", "tasty", "very", "and", ".", "!", "<END>", "<UNK>"
];

// --- Explanations Dictionary ---

const EXPLANATIONS = {
  tokenization: {
    title: "Tokenization",
    simple: "Breaking text into small pieces (tokens). Think of it like looking up words in a dictionary and getting their page number.",
    technical: "Text segmentation into subword units (tokens) using algorithms like BPE (Byte Pair Encoding) or WordPiece. Each unique token maps to a unique Integer ID from the model's fixed vocabulary table.",
    details: (
      <div className="space-y-2">
        <p><strong>How it works:</strong> The raw string is processed by a Tokenizer model (separate from the main LLM).</p>
        <ul className="list-disc pl-4 space-y-1">
          <li><strong>Input:</strong> Raw text string.</li>
          <li><strong>Algorithm:</strong> Ideally BPE/SentencePiece. This demo uses whitespace splitting for simplicity.</li>
          <li><strong>Lookup:</strong> Each unique string matches a row in the <code>Vocabulary</code> table.</li>
        </ul>
      </div>
    )
  },
  embedding: {
    title: "Embedding & Position",
    simple: "Converting numbers into rich meaning vectors. 'King' and 'Queen' will have similar vectors. We also add 'address' tags so the model knows word order.",
    technical: "Projects discrete Token IDs into a continuous vector space ($d_{model}$). Learned Positional Encodings (or Sinusoidal) are added element-wise to inject sequence order information, as Attention is permutation invariant.",
    details: (
      <div className="space-y-2">
        <p><strong>Why Embeddings?</strong> IDs like '5' and '6' have no semantic relationship. Vectors allow mathematical closeness (Dot Product) to represent semantic similarity.</p>
        <p><strong>Why Position?</strong> The Transformer processes all tokens in parallel. Without positional encodings, 'Cats eat fish' and 'Fish eat cats' would look identical to the self-attention mechanism.</p>
      </div>
    )
  },
  attention: {
    title: "Multi-Head Attention",
    simple: "The model looks at all previous words to understand context. It asks: 'Given what I am (Query), what other words (Keys) are relevant to me?'",
    technical: "Computes a weighted sum of Value vectors ($V$). Weights are determined by the compatibility of Query ($Q$) and Key ($K$) vectors via scaled dot-product attention: $\\text{softmax}(\\frac{QK^T}{\\sqrt{d_k}})V$.",
    details: (
      <div className="space-y-2">
        <p><strong>The Database Analogy:</strong></p>
        <ul className="list-disc pl-4 space-y-1">
          <li><strong>Query (Q):</strong> What I'm looking for.</li>
          <li><strong>Key (K):</strong> The tag/label of the information.</li>
          <li><strong>Value (V):</strong> The actual content/information.</li>
        </ul>
        <p>If $Q$ matches $K$ (high dot product), we take more of $V$.</p>
      </div>
    )
  },
  ffn: {
    title: "Feed Forward Network",
    simple: "Thinking time. The model processes the information it gathered from attention to build a deeper understanding.",
    technical: "A position-wise fully connected network applied to each token independently. Consists of two linear transformations with a non-linear activation (ReLU/GELU) in between: $W_2(\\text{ReLU}(W_1x + b_1)) + b_2$.",
    details: (
      <div className="space-y-2">
        <p><strong>Role:</strong> While Attention gathers information from <em>other</em> tokens, the FFN processes information within the <em>current</em> token's representation.</p>
        <p>It effectively acts as a key-value memory of facts learned during training.</p>
      </div>
    )
  },
  output: {
    title: "Unembedding & Sampling",
    simple: "Turning the final concept back into a list of word probabilities. We roll a weighted dice to pick the next word.",
    technical: "The final hidden state is projected back to Vocabulary size (logits). Softmax converts logits to a probability distribution. Sampling strategies (Greedy, Top-k, Nucleus) select the next token index.",
    details: (
      <div className="space-y-2">
        <p><strong>Logits:</strong> Raw scores (can be negative).</p>
        <p><strong>Softmax:</strong> Normalizes scores so they sum to 1.0.</p>
        <p><strong>Temperature:</strong> Divides logits before softmax. Low temp (&lt;1) exaggerates differences (confident); High temp (&gt;1) flattens distribution (random).</p>
      </div>
    )
  }
};

// --- Math Utilities ---

const generateMatrix = (rows: number, cols: number, seed: number = 1): number[][] => {
  const matrix: number[][] = [];
  for (let i = 0; i < rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < cols; j++) {
      const val = Math.sin(i * 10 + j * 20 + seed) * 0.5;
      row.push(val);
    }
    matrix.push(row);
  }
  return matrix;
};

const addVectors = (v1: number[], v2: number[]): number[] => {
  return v1.map((val, i) => val + (v2[i] || 0));
};

const matMulVector = (vec: number[], matrix: number[][]): number[] => {
  const result: number[] = [];
  const cols = matrix[0].length;
  for (let j = 0; j < cols; j++) {
    let sum = 0;
    for (let i = 0; i < vec.length; i++) {
      sum += vec[i] * matrix[i][j];
    }
    result.push(sum);
  }
  return result;
};

const dotProduct = (v1: number[], v2: number[]): number => {
  return v1.reduce((sum, val, i) => sum + val * v2[i], 0);
};

const softmax = (logits: number[], temperature: number = 1.0): number[] => {
  const maxLogit = Math.max(...logits);
  const exps = logits.map(l => Math.exp((l - maxLogit) / temperature));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sumExps);
};

const relu = (vec: number[]): number[] => vec.map(x => Math.max(0, x));

// --- Components ---

const Tooltip = ({ text, children }: { text: string, children: React.ReactNode }) => (
  <div className="group relative flex flex-col items-center">
    {children}
    <div className="absolute bottom-full mb-2 w-48 hidden group-hover:block z-50">
      <div className="bg-gray-900 text-white text-[10px] p-2 rounded shadow-lg">
        {text}
      </div>
      <div className="w-0 h-0 border-l-4 border-l-transparent border-r-4 border-r-transparent border-t-4 border-t-gray-900 mx-auto"></div>
    </div>
  </div>
);

const MatrixView = ({ data, title, labelsX, labelsY, highlightRow, colorClass = "bg-white", small = false }: any) => {
  const [expanded, setExpanded] = useState(false);
  
  if (!data || data.length === 0) return null;

  const matrix = Array.isArray(data[0]) ? data : [data];
  const rows = matrix.length;
  const cols = matrix[0].length;

  return (
    <div className={`border rounded bg-white shadow-sm overflow-hidden ${small ? 'text-[9px]' : 'text-xs'} font-mono my-1`}>
      <div 
        className={`flex justify-between items-center px-2 py-1 cursor-pointer bg-gray-50 hover:bg-gray-100 ${colorClass}`} 
        onClick={() => setExpanded(!expanded)}
      >
        <span className="font-bold text-gray-700 truncate">{title}</span>
        <span className="text-[9px] text-gray-400 ml-2">({rows}x{cols}) {expanded ? '▼' : '▶'}</span>
      </div>
      
      {expanded && (
        <div className="overflow-x-auto p-2">
          <table className="border-collapse w-full">
            {labelsX && (
              <thead>
                <tr>
                  {labelsY && <th></th>}
                  {labelsX.map((l: string, i: number) => <th key={i} className="px-1 text-gray-400 font-normal">{l}</th>)}
                </tr>
              </thead>
            )}
            <tbody>
              {matrix.map((row: number[], i: number) => (
                <tr key={i} className={highlightRow === i ? "bg-yellow-100" : ""}>
                  {labelsY && <td className="pr-2 text-gray-500 font-bold text-right">{labelsY[i]}</td>}
                  {row.map((val: number, j: number) => (
                    <td key={j} className="border border-gray-100 px-1 py-0.5 text-right min-w-[2.5rem]">
                      {val.toFixed(2)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

const DiagramBlock = ({ title, children, active = false, onClick, color = "gray", description, mode, explanationKey }: any) => {
  const colors: Record<string, string> = {
    gray: "border-gray-300 bg-gray-50",
    blue: "border-blue-400 bg-blue-50",
    purple: "border-purple-400 bg-purple-50",
    orange: "border-orange-400 bg-orange-50",
    green: "border-green-400 bg-green-50",
    red: "border-red-400 bg-red-50",
  };

  // Get explanation based on mode
  const text = explanationKey && EXPLANATIONS[explanationKey as keyof typeof EXPLANATIONS] 
    ? EXPLANATIONS[explanationKey as keyof typeof EXPLANATIONS][mode as 'simple' | 'technical']
    : description;

  return (
    <div 
      className={`relative border-2 rounded-lg p-3 transition-all duration-200 ${active ? 'ring-2 ring-offset-2 ring-blue-500 shadow-md' : 'hover:border-gray-400'} ${colors[color]} cursor-pointer group`}
      onClick={onClick}
    >
      <div className="flex justify-between items-start mb-2">
        <h3 className="font-bold text-sm uppercase tracking-wider text-gray-700 flex items-center gap-2">
          {title}
        </h3>
        <div className="flex items-center gap-1">
           {active && <span className="text-[10px] bg-blue-100 text-blue-800 px-1.5 rounded-full">Active</span>}
           <Info size={14} className="text-gray-400 hover:text-blue-600 transition-colors" />
        </div>
      </div>
      
      <p className="text-[11px] text-gray-600 mb-3 leading-snug font-medium opacity-90">
        {text}
      </p>

      <div className="space-y-2">
        {children}
      </div>
    </div>
  );
};

const ArrowDown = () => (
  <div className="flex justify-center my-1 text-gray-300">
    <ArrowRight className="rotate-90" size={20} />
  </div>
);

// --- Main Application ---

export default function App() {
  // --- State ---
  const [inputText, setInputText] = useState("I like");
  const [generatedTokens, setGeneratedTokens] = useState<string[]>([]);
  const [explanationMode, setExplanationMode] = useState<'simple' | 'technical'>('simple');
  
  // Params
  const [embedDim, setEmbedDim] = useState(4);
  const [temperature, setTemperature] = useState(1.0);
  const [maxTokens, setMaxTokens] = useState(5);
  
  // Simulation State
  const [pipelineData, setPipelineData] = useState<any>(null);
  const [isAutoRunning, setIsAutoRunning] = useState(false);
  const [activeStep, setActiveStep] = useState<string | null>(null); // For detail view
  
  const autoRunRef = useRef<NodeJS.Timeout | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // --- Derived Constants (Model Weights Simulation) ---
  const weights = useMemo(() => {
    return {
      wQ: generateMatrix(embedDim, embedDim, 1),
      wK: generateMatrix(embedDim, embedDim, 2),
      wV: generateMatrix(embedDim, embedDim, 3),
      wO: generateMatrix(embedDim, embedDim, 4),
      wFF1: generateMatrix(embedDim, embedDim * 2, 5),
      wFF2: generateMatrix(embedDim * 2, embedDim, 6),
      wVocab: generateMatrix(embedDim, VOCAB.length, 7),
      embeddings: generateMatrix(VOCAB.length, embedDim, 8),
      posEncodings: generateMatrix(20, embedDim, 9),
    };
  }, [embedDim]);

  // --- Logic ---

  const tokenize = (text: string) => {
    const rawTokens = text.trim().split(/\s+/);
    const validTokens = rawTokens.map(t => {
      const clean = t.replace(/[.,!]/g, "");
      return VOCAB.includes(clean) ? clean : "<UNK>";
    });
    return ["<START>", ...validTokens];
  };

  const runPipeline = (currentTokens: string[]) => {
    const contextSize = currentTokens.length;
    
    // 1. Tokenization to IDs
    const tokenIds = currentTokens.map(t => VOCAB.indexOf(t));

    // 2. Embedding + Positional Encoding
    const embeddings = tokenIds.map((id, pos) => {
      const emb = weights.embeddings[id];
      const posEnc = weights.posEncodings[pos];
      return {
        token: currentTokens[pos],
        base: emb,
        pos: posEnc,
        final: addVectors(emb, posEnc)
      };
    });

    const X = embeddings.map(e => e.final);

    // 3. Self-Attention
    const Q = X.map(vec => matMulVector(vec, weights.wQ));
    const K = X.map(vec => matMulVector(vec, weights.wK));
    const V = X.map(vec => matMulVector(vec, weights.wV));

    const attentionScores: number[][] = [];
    const attentionWeights: number[][] = [];

    for (let i = 0; i < contextSize; i++) {
      const rowScores: number[] = [];
      for (let j = 0; j < contextSize; j++) {
        if (j > i) {
          rowScores.push(-1e9); 
        } else {
          const score = dotProduct(Q[i], K[j]) / Math.sqrt(embedDim);
          rowScores.push(score);
        }
      }
      attentionScores.push(rowScores);
      attentionWeights.push(softmax(rowScores));
    }

    const attentionOutput = attentionWeights.map((weightsRow) => {
      const result = new Array(embedDim).fill(0);
      for (let j = 0; j < contextSize; j++) {
        const v_j = V[j];
        const w = weightsRow[j];
        for (let d = 0; d < embedDim; d++) {
          result[d] += w * v_j[d];
        }
      }
      return result;
    });

    // Residual (Simplified Add)
    const postAttention = attentionOutput.map((vec, i) => addVectors(vec, X[i]));

    // 4. Feed Forward Network
    const ffnHidden = postAttention.map(vec => relu(matMulVector(vec, weights.wFF1)));
    const ffnOutput = ffnHidden.map(vec => matMulVector(vec, weights.wFF2));
    
    // Residual #2
    const finalBlockOutput = ffnOutput.map((vec, i) => addVectors(vec, postAttention[i]));

    // 5. Output Head
    const lastState = finalBlockOutput[finalBlockOutput.length - 1];
    const logits = matMulVector(lastState, weights.wVocab);
    const probs = softmax(logits, temperature);

    return {
      tokens: currentTokens,
      embeddings,
      Q, K, V,
      attentionScores,
      attentionWeights,
      ffnHidden,
      finalBlockOutput,
      logits,
      probs
    };
  };

  const handleGenerateStep = () => {
    const fullSequence = [...tokenize(inputText), ...generatedTokens];
    if (fullSequence[fullSequence.length - 1] === "<END>") return;
    if (generatedTokens.length >= maxTokens) return;

    const results = runPipeline(fullSequence);
    setPipelineData(results);

    // Sample next token
    let nextTokenIndex = 0;
    const r = Math.random();
    let accumulated = 0;
    
    for (let i = 0; i < results.probs.length; i++) {
      accumulated += results.probs[i];
      if (r <= accumulated) {
        nextTokenIndex = i;
        break;
      }
    }

    const nextToken = VOCAB[nextTokenIndex];
    setGeneratedTokens(prev => [...prev, nextToken]);
    
    // Auto scroll to bottom of diagram
    setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 100);
  };

  const handleRunFull = () => {
    setGeneratedTokens([]);
    setPipelineData(null);
    setActiveStep(null);
    // Initial run
    setTimeout(() => {
        const initialTokens = tokenize(inputText);
        const results = runPipeline(initialTokens);
        setPipelineData(results);
    }, 100);
  };

  const toggleAutoRun = () => {
    if (isAutoRunning) {
      clearInterval(autoRunRef.current!);
      setIsAutoRunning(false);
    } else {
      setIsAutoRunning(true);
    }
  };

  useEffect(() => {
    if (isAutoRunning) {
      autoRunRef.current = setInterval(() => {
        handleGenerateStep();
      }, 1500);
    }
    return () => clearInterval(autoRunRef.current!);
  }, [isAutoRunning, generatedTokens, maxTokens]);

  useEffect(() => {
    const lastToken = generatedTokens[generatedTokens.length - 1];
    if (generatedTokens.length >= maxTokens || lastToken === "<END>") {
      setIsAutoRunning(false);
      clearInterval(autoRunRef.current!);
    }
  }, [generatedTokens, maxTokens]);


  // --- Render Helpers ---

  const renderWeightsIcon = (label: string, tooltip: string) => (
    <Tooltip text={tooltip}>
      <div className="flex flex-col items-center justify-center p-1 bg-gray-100 border border-gray-300 hover:border-blue-400 hover:bg-blue-50 transition-colors rounded text-[9px] font-mono text-gray-500 w-12 h-10 mx-2 shadow-sm cursor-help">
         <Database size={10} className="mb-1"/>
         {label}
      </div>
    </Tooltip>
  );

  return (
    <div className="h-screen bg-gray-100 text-gray-800 font-sans flex flex-col md:flex-row overflow-hidden">
      
      {/* --- LEFT PANEL: Controls --- */}
      <div className="w-full md:w-80 bg-white border-r border-gray-200 p-5 flex flex-col gap-5 shadow-lg z-20 overflow-y-auto h-full flex-shrink-0">
        
        <div className="flex flex-col gap-2">
          <h1 className="text-xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-blue-700 to-indigo-600">
            LLM Internals
          </h1>
          
          {/* View Toggle */}
          <div className="bg-gray-100 p-1 rounded-lg flex text-xs font-bold">
            <button 
              onClick={() => setExplanationMode('simple')}
              className={`flex-1 py-1.5 rounded-md flex items-center justify-center gap-1 transition-all ${explanationMode === 'simple' ? 'bg-white text-blue-600 shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}
            >
              <Brain size={14} /> Intuitive
            </button>
            <button 
              onClick={() => setExplanationMode('technical')}
              className={`flex-1 py-1.5 rounded-md flex items-center justify-center gap-1 transition-all ${explanationMode === 'technical' ? 'bg-white text-purple-600 shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}
            >
              <Calculator size={14} /> Technical
            </button>
          </div>
        </div>

        {/* Input Section */}
        <div className="space-y-2">
          <label className="text-xs font-bold uppercase tracking-wider text-gray-500 block">Prompt</label>
          <div className="relative">
             <textarea 
               className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none text-sm shadow-inner"
               rows={2}
               value={inputText}
               onChange={(e) => setInputText(e.target.value)}
             />
          </div>
          <div className="flex gap-2 text-[10px] flex-wrap">
             {["I like", "code is", "cats are"].map(s => (
               <button key={s} onClick={() => { setInputText(s); handleRunFull(); }} className="px-2 py-1 bg-gray-100 hover:bg-gray-200 rounded border transition-colors">
                 "{s}"
               </button>
             ))}
          </div>
        </div>

        {/* Parameters */}
        <div className="space-y-4 border-t pt-4">
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-gray-500">
            <Settings size={14} /> Hyperparameters
          </div>
          
          <div className="space-y-3">
             <div>
                <div className="flex justify-between text-xs mb-1">
                  <span>Temperature</span>
                  <span className="font-mono bg-gray-100 px-1 rounded">{temperature}</span>
                </div>
                <input 
                  type="range" min="0.1" max="2.0" step="0.1" 
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full accent-blue-600 h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
             </div>

             <div>
                <div className="flex justify-between text-xs mb-1">
                  <span>Embedding Size</span>
                  <span className="font-mono bg-gray-100 px-1 rounded">{embedDim}</span>
                </div>
                <input 
                  type="range" min="2" max="6" step="1" 
                  value={embedDim}
                  onChange={(e) => {
                     setEmbedDim(parseInt(e.target.value));
                     setPipelineData(null); 
                     setGeneratedTokens([]);
                  }}
                  className="w-full accent-indigo-600 h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
             </div>
             
             <div>
                <div className="flex justify-between text-xs mb-1">
                  <span>Max Tokens</span>
                  <span className="font-mono bg-gray-100 px-1 rounded">{maxTokens}</span>
                </div>
                <input 
                  type="range" min="1" max="10" step="1" 
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                  className="w-full accent-green-600 h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
             </div>
          </div>
        </div>

        {/* Detail View Panel (When user clicks a block) */}
        {activeStep && (
           <div className="mt-2 border-t pt-4 animate-in slide-in-from-bottom-5 fade-in duration-300">
             <div className="flex justify-between items-center mb-2">
               <h3 className="font-bold text-sm text-blue-800">
                 {EXPLANATIONS[activeStep as keyof typeof EXPLANATIONS]?.title || activeStep}
               </h3>
               <button onClick={() => setActiveStep(null)} className="text-gray-400 hover:text-gray-600"><X size={14}/></button>
             </div>
             
             <div className="bg-blue-50/50 rounded p-3 text-xs border border-blue-100 mb-2 leading-relaxed">
               {EXPLANATIONS[activeStep as keyof typeof EXPLANATIONS]?.details}
             </div>

             {/* Live Data Visuals in Sidebar */}
             {pipelineData && (
                <div className="bg-gray-50 rounded p-2 text-xs border overflow-y-auto max-h-48 shadow-inner">
                  {activeStep === "tokenization" && (
                    <div className="space-y-2">
                      <p className="text-gray-500">Vocabulary Lookup Table:</p>
                      <div className="grid grid-cols-2 gap-1 font-mono text-[10px] bg-white p-2 rounded border">
                         {VOCAB.slice(0, 8).map((v, i) => (
                           <div key={i} className="flex justify-between border-b border-dashed"><span>"{v}"</span> <span className="text-gray-400">{i}</span></div>
                         ))}
                         <div className="text-center col-span-2 text-gray-400">...</div>
                      </div>
                    </div>
                  )}
                  {activeStep === "embedding" && (
                    <>
                       <MatrixView title="Current Input X" data={pipelineData.embeddings.map((e:any) => e.final)} highlightRow={pipelineData.tokens.length-1} />
                    </>
                  )}
                  {activeStep === "attention" && (
                    <>
                       <MatrixView title="Attention Scores" data={pipelineData.attentionScores} small />
                       <MatrixView title="Attn Weights (Softmax)" data={pipelineData.attentionWeights} small />
                    </>
                  )}
                  {activeStep === "ffn" && (
                    <>
                       <MatrixView title="Hidden State (ReLU)" data={pipelineData.ffnHidden} small />
                    </>
                  )}
                  {activeStep === "output" && (
                    <>
                       <MatrixView title="Logits (Scores)" data={pipelineData.logits} small />
                    </>
                  )}
                </div>
             )}
           </div>
        )}

        {/* Actions */}
        <div className="mt-auto pt-4 flex flex-col gap-2 bg-white sticky bottom-0">
          <button 
            onClick={handleRunFull}
            className="w-full py-2 bg-gray-800 hover:bg-gray-900 text-white rounded shadow flex items-center justify-center gap-2 font-medium transition-all"
          >
            <RefreshCw size={16} /> Reset & Start
          </button>
          
          <div className="grid grid-cols-2 gap-2">
            <button 
              onClick={handleGenerateStep}
              disabled={generatedTokens.length >= maxTokens || isAutoRunning}
              className="py-2 bg-white hover:bg-gray-50 text-gray-700 border border-gray-300 rounded shadow-sm flex items-center justify-center gap-2 disabled:opacity-50 transition-all"
            >
              <SkipForward size={16} /> Step
            </button>
            <button 
              onClick={toggleAutoRun}
              disabled={generatedTokens.length >= maxTokens}
              className={`py-2 border rounded shadow-sm flex items-center justify-center gap-2 disabled:opacity-50 transition-all ${isAutoRunning ? 'bg-red-50 text-red-600 border-red-200' : 'bg-green-50 text-green-700 border-green-200 hover:bg-green-100'}`}
            >
               {isAutoRunning ? "Stop" : <><Play size={16} /> Auto</>}
            </button>
          </div>
        </div>

      </div>

      {/* --- RIGHT PANEL: Flowchart Canvas --- */}
      <div className="flex-1 overflow-y-auto overflow-x-hidden bg-slate-50 relative">
        <div className="min-h-full p-8 flex flex-col items-center max-w-4xl mx-auto">
          
          {/* Header Sequence */}
          <div className="w-full bg-white p-4 rounded-xl shadow-sm border border-slate-200 mb-8 sticky top-0 z-10 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
             <div className="flex-1">
                <h2 className="text-xs uppercase tracking-wide font-bold text-gray-400 mb-2">Context Window</h2>
                <div className="flex flex-wrap gap-2 items-center">
                  {tokenize(inputText).map((t, i) => (
                    <div key={`input-${i}`} className="flex flex-col items-center group">
                        <span className="px-3 py-1 rounded-md bg-blue-100 text-blue-800 border border-blue-200 text-sm font-mono shadow-sm group-hover:ring-2 ring-blue-300 transition-all cursor-default" title={`Token ID: ${VOCAB.indexOf(t)}`}>{t}</span>
                        <span className="text-[9px] text-gray-400 mt-0.5">{i}</span>
                    </div>
                  ))}
                  {generatedTokens.map((t, i) => (
                    <div key={`gen-${i}`} className="flex flex-col items-center animate-in zoom-in duration-300 group">
                        <span className={`px-3 py-1 rounded-md text-sm font-mono shadow-sm border group-hover:ring-2 ring-green-300 transition-all cursor-default ${i === generatedTokens.length - 1 ? 'bg-green-100 text-green-800 border-green-300 ring-2 ring-green-400 ring-offset-1' : 'bg-green-50 text-green-700 border-green-200'}`} title={`Token ID: ${VOCAB.indexOf(t)}`}>
                          {t}
                        </span>
                        <span className="text-[9px] text-gray-400 mt-0.5">{tokenize(inputText).length + i}</span>
                    </div>
                  ))}
                  {generatedTokens.length < maxTokens && (
                      <div className="w-8 h-8 rounded border-2 border-dashed border-gray-300 flex items-center justify-center text-gray-300 text-xs">?</div>
                  )}
                </div>
             </div>
             
             {/* Autoregressive Arrow Visual */}
             {generatedTokens.length > 0 && (
                <div className="hidden md:flex items-center text-green-600 text-xs font-bold gap-2 bg-green-50 px-3 py-2 rounded-lg border border-green-100 animate-pulse">
                   <CornerDownLeft size={16} />
                   <span>Looping back</span>
                </div>
             )}
          </div>

          {!pipelineData ? (
             <div className="flex-1 flex flex-col items-center justify-center text-gray-400 mt-20">
               <div className="p-6 bg-white rounded-full shadow-lg mb-6 animate-bounce">
                  <Play size={32} className="text-blue-500 ml-1" />
               </div>
               <h3 className="text-lg font-bold text-gray-600">Ready to Simulate</h3>
               <p className="text-sm">Click "Reset & Start" or "Step" to begin the transformer pipeline.</p>
             </div>
          ) : (
            <div className="w-full max-w-2xl relative">
              
              {/* FLOWCHART START */}

               {/* 0. TOKENIZATION STAGE (NEW) */}
              <div className="relative z-0 mb-8">
                  <DiagramBlock 
                    title="0. Tokenization" 
                    color="gray" 
                    mode={explanationMode}
                    explanationKey="tokenization"
                    active={activeStep === "tokenization"}
                    onClick={() => setActiveStep("tokenization")}
                  >
                    <div className="flex items-center justify-center gap-4 py-2">
                       <div className="flex flex-col items-center bg-white border border-dashed border-gray-300 rounded p-2">
                           <span className="font-serif italic text-gray-600">"I like cats"</span>
                           <span className="text-[9px] text-gray-400">Raw Text</span>
                       </div>
                       <ArrowRight size={16} className="text-gray-400" />
                       <div className="flex flex-col items-center bg-white border border-gray-200 rounded p-2 shadow-sm">
                           <List size={16} className="text-blue-500 mb-1" />
                           <span className="text-[9px] font-bold text-gray-600">Tokenizer</span>
                       </div>
                       <ArrowRight size={16} className="text-gray-400" />
                       <div className="flex gap-1">
                          {[1, 2, 5].map(n => <span key={n} className="bg-gray-100 border px-1.5 rounded font-mono text-xs">{n}</span>)}
                       </div>
                    </div>
                  </DiagramBlock>
                  <ArrowDown />
              </div>

              {/* 1. INPUT STAGE */}
              <div className="relative z-0">
                  <DiagramBlock 
                    title="1. Embedding Layer" 
                    color="blue" 
                    mode={explanationMode}
                    explanationKey="embedding"
                    active={activeStep === "embedding"}
                    onClick={() => setActiveStep("embedding")}
                  >
                    <div className="flex items-center justify-center gap-4">
                       <div className="text-center">
                          <div className="text-[10px] text-gray-500 mb-1">IDs</div>
                          <div className="bg-white border px-2 py-1 rounded text-xs font-mono shadow-sm">Token IDs</div>
                       </div>
                       <ArrowRight size={16} className="text-gray-400" />
                       <div className="flex flex-col items-center bg-white border border-blue-200 rounded p-2 shadow-sm relative group">
                           <Search size={14} className="text-blue-500 mb-1" />
                           <span className="text-[9px] font-bold text-gray-600">Lookup</span>
                       </div>
                       <ArrowRight size={16} className="text-gray-400" />
                       <div className="text-center">
                          <div className="text-[10px] text-gray-500 mb-1">Vectors</div>
                          <div className="bg-indigo-50 border border-indigo-200 px-2 py-1 rounded text-xs font-mono text-indigo-800">
                             Vectors (X)
                          </div>
                       </div>
                    </div>
                  </DiagramBlock>
              </div>

              <ArrowDown />

              {/* 2. TRANSFORMER BLOCK (CONTAINER) */}
              <div className="border-2 border-dashed border-slate-300 rounded-xl p-4 bg-slate-50/50 relative">
                  <span className="absolute -top-3 left-4 bg-slate-50 px-2 text-xs font-bold text-slate-500 uppercase tracking-wider">Transformer Block</span>
                  
                  {/* SELF ATTENTION */}
                  <DiagramBlock 
                    title="2. Multi-Head Attention" 
                    color="purple"
                    mode={explanationMode}
                    explanationKey="attention"
                    active={activeStep === "attention"}
                    onClick={() => setActiveStep("attention")}
                  >
                    <div className="flex items-center justify-between">
                       {/* Weights Inputs */}
                       <div className="flex flex-col gap-1 mr-2">
                          {renderWeightsIcon("W_Q", "Query Weights: Transforms input into 'Search Query'")}
                          {renderWeightsIcon("W_K", "Key Weights: Transforms input into 'Search Tags'")}
                          {renderWeightsIcon("W_V", "Value Weights: Transforms input into 'Content'")}
                       </div>
                       
                       {/* Operation */}
                       <div className="flex-1 bg-white rounded border border-purple-200 p-2 flex flex-col items-center shadow-sm">
                          <div className="flex gap-4 mb-2">
                             <Tooltip text="Query: What I am looking for">
                               <span className="bg-purple-100 text-purple-800 px-2 rounded text-[10px] font-bold cursor-help">Q</span>
                             </Tooltip>
                             <Tooltip text="Key: What I contain">
                               <span className="bg-purple-100 text-purple-800 px-2 rounded text-[10px] font-bold cursor-help">K</span>
                             </Tooltip>
                             <Tooltip text="Value: What information I pass on">
                               <span className="bg-purple-100 text-purple-800 px-2 rounded text-[10px] font-bold cursor-help">V</span>
                             </Tooltip>
                          </div>
                          <div className="text-xs font-mono mb-1 text-gray-500">Softmax(QKᵀ / √d) · V</div>
                          
                          {/* Mini Heatmap Visualization */}
                          <div className="mt-2 w-full flex justify-center">
                             <div className="grid grid-cols-5 gap-0.5 opacity-80">
                                {pipelineData.attentionWeights[pipelineData.attentionWeights.length-1]?.map((w: number, idx: number) => (
                                   <div key={idx} className="w-4 h-4" style={{ backgroundColor: `rgba(147, 51, 234, ${w})` }}></div>
                                ))}
                             </div>
                          </div>
                          <span className="text-[9px] text-gray-400 mt-1">Attention Map (Last Token)</span>
                       </div>
                    </div>
                  </DiagramBlock>

                  <ArrowDown />
                  
                  <div className="flex justify-center mb-1">
                     <Tooltip text="Residual Connection & Layer Norm: Adds stability to training.">
                       <span className="bg-gray-200 text-gray-600 text-[9px] px-2 rounded-full cursor-help">Add & Norm</span>
                     </Tooltip>
                  </div>

                  <ArrowDown />

                  {/* FEED FORWARD */}
                  <DiagramBlock 
                    title="3. Feed Forward Network" 
                    color="orange"
                    mode={explanationMode}
                    explanationKey="ffn"
                    active={activeStep === "ffn"}
                    onClick={() => setActiveStep("ffn")}
                  >
                     <div className="flex items-center justify-between">
                        <div className="flex flex-col gap-1 mr-2">
                          {renderWeightsIcon("W_FF1", "First Linear Layer (Expansion)")}
                          {renderWeightsIcon("W_FF2", "Second Linear Layer (Projection)")}
                        </div>

                        <div className="flex-1 flex items-center justify-center gap-2">
                           <div className="bg-orange-50 border border-orange-200 p-2 rounded text-center">
                              <div className="text-xs font-bold text-orange-800">Linear</div>
                              <div className="text-[9px] text-orange-600">Expand</div>
                           </div>
                           <ArrowRight size={12} />
                           <div className="bg-white border px-2 py-1 rounded shadow-sm" title="Activation Function (ReLU/GELU)">
                              <div className="text-xs font-bold">ReLU</div>
                           </div>
                           <ArrowRight size={12} />
                           <div className="bg-orange-50 border border-orange-200 p-2 rounded text-center">
                              <div className="text-xs font-bold text-orange-800">Linear</div>
                              <div className="text-[9px] text-orange-600">Project</div>
                           </div>
                        </div>
                     </div>
                  </DiagramBlock>

                  <ArrowDown />

                  <div className="flex justify-center">
                     <Tooltip text="Residual Connection & Layer Norm">
                       <span className="bg-gray-200 text-gray-600 text-[9px] px-2 rounded-full cursor-help">Add & Norm</span>
                     </Tooltip>
                  </div>
              </div>

              <ArrowDown />

              {/* 4. OUTPUT STAGE */}
              <div className="relative z-0">
                  <DiagramBlock 
                    title="4. Unembedding & Sampling" 
                    color="green"
                    mode={explanationMode}
                    explanationKey="output"
                    active={activeStep === "output"}
                    onClick={() => setActiveStep("output")}
                  >
                    <div className="flex flex-col gap-4">
                       {/* Logits Calculation */}
                       <div className="flex items-center justify-center gap-4 border-b border-green-100 pb-2">
                          {renderWeightsIcon("W_Vocab", "Vocabulary Projection Matrix")}
                          <div className="text-lg text-gray-400">×</div>
                          <div className="bg-orange-50 border border-orange-200 px-2 py-1 rounded text-xs text-orange-800">Final Vector</div>
                          <ArrowRight size={16} />
                          <div className="bg-green-50 border border-green-200 px-2 py-1 rounded text-xs text-green-800 font-bold">Logits</div>
                       </div>

                       {/* Softmax & Sampling Visualization */}
                       <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          {/* Chart */}
                          <div className="h-24 flex items-end gap-0.5 border-b border-gray-300">
                             {pipelineData.probs.map((p: number, i: number) => (
                                <div key={i} className="flex-1 bg-green-400 hover:bg-green-600 transition-colors relative group" style={{ height: `${p*100}%` }}>
                                   <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block bg-black text-white text-[9px] px-1 rounded whitespace-nowrap z-50">
                                      {VOCAB[i]}: {(p*100).toFixed(0)}%
                                   </div>
                                </div>
                             ))}
                          </div>
                          
                          {/* Selection Logic */}
                          <div className="flex flex-col justify-center text-xs space-y-2">
                             <div className="flex justify-between items-center bg-gray-50 p-1 rounded">
                                <span>Temperature:</span>
                                <span className="font-mono font-bold">{temperature}</span>
                             </div>
                             <div className="flex justify-between items-center bg-yellow-50 p-1 rounded border border-yellow-200">
                                <span>Selected:</span>
                                <span className="font-mono font-bold text-yellow-800">
                                   {generatedTokens.length > 0 ? generatedTokens[generatedTokens.length-1] : "..."}
                                </span>
                             </div>
                             <p className="text-[9px] text-gray-400 leading-tight">
                                {explanationMode === 'simple' 
                                  ? "We spin a wheel of fortune where higher bars have bigger slices." 
                                  : "Token sampled from probability distribution (Categorical)."}
                             </p>
                          </div>
                       </div>
                    </div>
                  </DiagramBlock>
              </div>

              {/* LOOP VISUALIZATION LINE */}
              <div className="absolute top-10 -right-4 md:-right-12 bottom-20 w-8 md:w-12 border-r-2 border-b-2 border-dashed border-gray-300 rounded-br-3xl pointer-events-none opacity-50 hidden md:block" />
              <div className="absolute bottom-20 -right-4 md:-right-12 text-gray-300 text-[10px] rotate-90 hidden md:block">Autoregressive Loop</div>

              {/* Bottom Anchor */}
              <div ref={bottomRef} className="h-4" />

            </div>
          )}
        </div>
      </div>
    </div>
  );
}
