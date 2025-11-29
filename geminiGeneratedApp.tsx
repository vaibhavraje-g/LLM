import React, { useState, useEffect, useRef } from 'react';
import { 
  Play, Pause, RefreshCw, Activity, 
  TrendingDown, Brain, Zap, Target, 
  ArrowRight, BookOpen, GraduationCap,
  Save
} from 'lucide-react';

// --- Configuration ---
const LEARNING_RATE_DEFAULT = 0.1;
const VOCAB = ["Paris", "France", "Rome", "Italy", "is", "capital", "of", "."];
// Simple training data: "Paris is capital of France", "Rome is capital of Italy"
const TRAINING_DATA = [
  { input: "Paris", target: "is" },
  { input: "is", target: "capital" },
  { input: "capital", target: "of" },
  { input: "of", target: "France" }, // Knowledge: Paris -> France connection via context
  { input: "Rome", target: "is" },
  { input: "of", target: "Italy" }
];

// Map words to indices
const TOKEN_MAP = VOCAB.reduce((acc, word, i) => ({ ...acc, [word]: i }), {} as Record<string, number>);

// --- Math Helpers ---

// Generate random weights matrix (Vocab x Vocab for simple Bigram-ish logic)
// In reality, LLMs use Embedding -> Attention -> FFN -> Unembedding. 
// We simulate the "Knowledge Storage" which mostly happens in FFN/Projections.
const initWeights = (size: number) => {
  const w = [];
  for (let i = 0; i < size; i++) {
    const row = [];
    for (let j = 0; j < size; j++) {
      row.push(Math.random() * 0.2 - 0.1); // Small random values
    }
    w.push(row);
  }
  return w;
};

const softmax = (logits: number[]) => {
  const max = Math.max(...logits);
  const exps = logits.map(l => Math.exp(l - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
};

// --- Components ---

const ExplanationCard = ({ title, content, type = "info" }: { title: string, content: string, type?: "info" | "concept" }) => (
  <div className={`p-4 rounded-lg border mb-4 ${type === 'concept' ? 'bg-amber-50 border-amber-200' : 'bg-blue-50 border-blue-200'}`}>
    <h3 className={`font-bold text-sm mb-1 flex items-center gap-2 ${type === 'concept' ? 'text-amber-800' : 'text-blue-800'}`}>
      {type === 'concept' ? <GraduationCap size={16} /> : <BookOpen size={16} />}
      {title}
    </h3>
    <p className="text-xs text-gray-700 leading-relaxed">{content}</p>
  </div>
);

export default function LLMTrainingSimulator() {
  // --- State ---
  const [weights, setWeights] = useState<number[][]>(() => initWeights(VOCAB.length));
  const [isTraining, setIsTraining] = useState(false);
  const [step, setStep] = useState(0);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [currentPair, setCurrentPair] = useState<{input: string, target: string} | null>(null);
  const [currentPreds, setCurrentPreds] = useState<number[]>([]);
  const [learningRate, setLearningRate] = useState(LEARNING_RATE_DEFAULT);
  
  // Test Mode
  const [testInput, setTestInput] = useState("Paris");
  const [testOutput, setTestOutput] = useState("");

  const trainingRef = useRef<NodeJS.Timeout | null>(null);

  // --- Logic ---

  const performTrainingStep = () => {
    setWeights(prevWeights => {
      const dataIdx = step % TRAINING_DATA.length;
      const { input, target } = TRAINING_DATA[dataIdx];
      setCurrentPair({ input, target });

      const inputIdx = TOKEN_MAP[input];
      const targetIdx = TOKEN_MAP[target];

      // 1. Forward Pass (Simple lookup row in our simplified model)
      // In a real LLM, this is huge matrix multiplication.
      // Here, Weights[inputIdx] effectively acts as the "Logits" for the next token.
      const logits = [...prevWeights[inputIdx]]; 
      const probs = softmax(logits);
      setCurrentPreds(probs);

      // 2. Calculate Loss (Cross Entropy)
      // Loss = -ln(probability of correct target)
      const targetProb = probs[targetIdx];
      const loss = -Math.log(targetProb + 1e-10); // epsilon for stability

      // 3. Backpropagation (Gradients)
      // For Softmax + CrossEntropy, gradient wrt logits is (probs - ground_truth)
      // We update the row corresponding to the input token.
      const newWeights = prevWeights.map(row => [...row]);
      const gradients = probs.map((p, i) => {
        const truth = i === targetIdx ? 1 : 0;
        return p - truth; 
      });

      // 4. Optimization Step (Gradient Descent)
      // W_new = W_old - LearningRate * Gradient
      for (let i = 0; i < VOCAB.length; i++) {
        // We only update the weights connected to our input in this simplified bigram view
        // In real MLP, backprop goes through all layers.
        newWeights[inputIdx][i] -= learningRate * gradients[i]; 
      }

      setLossHistory(prev => [...prev.slice(-49), loss]);
      setStep(s => s + 1);
      
      return newWeights;
    });
  };

  useEffect(() => {
    if (isTraining) {
      trainingRef.current = setInterval(performTrainingStep, 200); // Speed of simulation
    } else {
      if (trainingRef.current) clearInterval(trainingRef.current);
    }
    return () => { if (trainingRef.current) clearInterval(trainingRef.current); };
  }, [isTraining, step]);

  const reset = () => {
    setIsTraining(false);
    setWeights(initWeights(VOCAB.length));
    setStep(0);
    setLossHistory([]);
    setCurrentPair(null);
    setCurrentPreds([]);
  };

  const predict = (word: string) => {
    const idx = TOKEN_MAP[word];
    if (idx === undefined) return "???";
    
    const row = weights[idx];
    const probs = softmax(row);
    
    // Greedy sampling
    let maxIdx = 0;
    for(let i=1; i<probs.length; i++) {
      if(probs[i] > probs[maxIdx]) maxIdx = i;
    }
    return VOCAB[maxIdx];
  };

  useEffect(() => {
    setTestOutput(predict(testInput));
  }, [testInput, weights]);

  // --- Rendering Helpers ---

  // Visualizing the "Brain" (Weight Matrix) as a heatmap
  const renderWeightMatrix = () => {
    return (
      <div className="grid grid-cols-9 gap-0.5 bg-gray-200 p-1 rounded border">
        {/* Header Row */}
        <div className="bg-gray-100 text-[8px] flex items-center justify-center font-bold text-gray-400">IN\OUT</div>
        {VOCAB.map(w => (
          <div key={w} className="bg-gray-100 text-[8px] flex items-center justify-center font-bold text-gray-500 overflow-hidden">{w.slice(0,3)}</div>
        ))}

        {/* Rows */}
        {VOCAB.map((inWord, i) => (
          <React.Fragment key={inWord}>
            {/* Row Label */}
            <div className={`text-[8px] flex items-center justify-end pr-1 font-bold ${currentPair?.input === inWord ? 'text-blue-600 bg-blue-50' : 'text-gray-500 bg-gray-50'}`}>
              {inWord}
            </div>
            
            {/* Cells */}
            {weights[i].map((val, j) => {
              // Normalize value for color roughly between -2 and 2
              const intensity = Math.min(Math.max((val + 1) / 2, 0), 1); 
              // Blue for positive (strong connection), Red for negative (inhibition)
              // Actually let's use Grayscale to Green for "Strength" concept simplicity
              // Or better: Heatmap style. High = Green, Low = Red.
              const isTarget = currentPair?.input === inWord && VOCAB[j] === currentPair?.target;
              
              let bgColor = "rgb(243 244 246)"; // gray-100
              if (val > 0.5) bgColor = `rgba(34, 197, 94, ${Math.min(val, 1)})`; // Greenish
              else if (val < -0.5) bgColor = `rgba(239, 68, 68, ${Math.min(Math.abs(val), 1)})`; // Reddish
              else bgColor = `rgba(209, 213, 219, ${0.3 + Math.abs(val)})`; // Neutral grays

              return (
                <div 
                  key={`${i}-${j}`} 
                  className={`h-6 w-full flex items-center justify-center text-[7px] transition-colors duration-200 ${isTarget ? 'ring-1 ring-blue-500 z-10' : ''}`}
                  style={{ backgroundColor: bgColor }}
                  title={`${inWord} -> ${VOCAB[j]}: ${val.toFixed(2)}`}
                >
                  {val.toFixed(1)}
                </div>
              );
            })}
          </React.Fragment>
        ))}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50 font-sans text-gray-800 p-4 md:p-8 flex flex-col gap-6">
      
      {/* Header */}
      <header className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b pb-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
            <Brain className="text-blue-600" />
            LLM Training Simulator
          </h1>
          <p className="text-sm text-gray-500">Visualize how weights change via Gradient Descent to store knowledge.</p>
        </div>
        <div className="flex gap-2">
          <button 
            onClick={() => setIsTraining(!isTraining)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-bold shadow-sm transition-all ${isTraining ? 'bg-amber-100 text-amber-700 hover:bg-amber-200' : 'bg-green-600 text-white hover:bg-green-700'}`}
          >
            {isTraining ? <><Pause size={18} /> Pause Training</> : <><Play size={18} /> Start Training</>}
          </button>
          <button 
            onClick={reset}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 shadow-sm"
          >
            <RefreshCw size={18} /> Reset
          </button>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* LEFT COLUMN: Concepts & Status */}
        <div className="space-y-6">
          
          <div className="bg-white p-5 rounded-xl shadow-sm border border-gray-200">
            <h2 className="font-bold text-gray-800 mb-4 flex items-center gap-2">
              <Activity size={18} /> Training Status
            </h2>
            
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="bg-gray-50 p-3 rounded border">
                <div className="text-xs text-gray-500 uppercase">Step (Epochs)</div>
                <div className="text-2xl font-mono">{step}</div>
              </div>
              <div className="bg-gray-50 p-3 rounded border">
                <div className="text-xs text-gray-500 uppercase">Current Loss</div>
                <div className={`text-2xl font-mono ${lossHistory[lossHistory.length-1] < 0.5 ? 'text-green-600' : 'text-red-500'}`}>
                  {lossHistory.length > 0 ? lossHistory[lossHistory.length-1].toFixed(4) : "N/A"}
                </div>
              </div>
            </div>

            <div className="mb-4">
               <label className="text-xs font-bold text-gray-500 mb-1 block">Learning Rate (Step Size)</label>
               <input 
                 type="range" min="0.01" max="0.5" step="0.01"
                 value={learningRate}
                 onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                 className="w-full accent-blue-600"
               />
               <div className="flex justify-between text-xs text-gray-400 mt-1">
                 <span>Cautious (0.01)</span>
                 <span>{learningRate}</span>
                 <span>Aggressive (0.5)</span>
               </div>
            </div>

            <div className="h-32 flex items-end gap-0.5 bg-gray-50 border rounded p-2 relative">
              {lossHistory.length === 0 && <div className="absolute inset-0 flex items-center justify-center text-xs text-gray-400">Loss graph will appear here</div>}
              {lossHistory.map((val, i) => (
                <div 
                  key={i} 
                  className="bg-red-400 w-full rounded-t"
                  style={{ height: `${Math.min(val * 20, 100)}%`, opacity: 0.5 + (i/lossHistory.length)*0.5 }}
                />
              ))}
            </div>
            <p className="text-xs text-gray-400 mt-1 text-center">Loss over time (Lower is better)</p>
          </div>

          <ExplanationCard 
            title="What is Training?" 
            type="concept"
            content="Training is the process of adjusting the 'weights' (connections) between neurons. Initially, the model guesses randomly (high loss). We calculate how wrong it is, and use math (Backpropagation) to nudge the weights slightly to reduce that error." 
          />
          
          <ExplanationCard 
            title="Interview Tip: Gradient Descent" 
            type="concept"
            content="Imagine being on a foggy mountain (the loss landscape) trying to find the lowest valley (zero error). You feel the slope with your feet (gradient) and take a step downhill. The Learning Rate is how big that step is." 
          />

        </div>

        {/* MIDDLE COLUMN: Visualization */}
        <div className="space-y-6">
          
          {/* Training Loop Visual */}
          <div className="bg-white p-5 rounded-xl shadow-sm border border-gray-200">
            <h2 className="font-bold text-gray-800 mb-4 flex items-center gap-2">
              <Zap size={18} className="text-amber-500" /> The Loop
            </h2>

            <div className="flex flex-col gap-2">
              {/* Step 1: Data */}
              <div className={`p-3 rounded border transition-all ${currentPair ? 'bg-blue-50 border-blue-300' : 'bg-gray-50 border-gray-200'}`}>
                <div className="text-xs font-bold text-gray-500 mb-1">1. Input Data Batch</div>
                <div className="flex items-center justify-center gap-2 font-mono text-sm">
                   {currentPair ? (
                     <>
                       <span className="bg-white px-2 py-1 rounded border shadow-sm">{currentPair.input}</span>
                       <ArrowRight size={14} className="text-gray-400"/>
                       <span className="bg-green-100 text-green-800 px-2 py-1 rounded border border-green-200 shadow-sm">{currentPair.target}</span>
                     </>
                   ) : "Waiting..."}
                </div>
              </div>

              <div className="flex justify-center"><ArrowRight className="rotate-90 text-gray-300" size={20}/></div>

              {/* Step 2: Prediction */}
              <div className={`p-3 rounded border transition-all ${currentPreds.length > 0 ? 'bg-purple-50 border-purple-300' : 'bg-gray-50 border-gray-200'}`}>
                 <div className="text-xs font-bold text-gray-500 mb-1">2. Model Prediction (Forward Pass)</div>
                 {currentPreds.length > 0 && currentPair ? (
                   <div className="space-y-1">
                     {/* Show top 3 preds */}
                     {currentPreds.map((p, i) => ({p, w: VOCAB[i]})).sort((a,b)=>b.p-a.p).slice(0,3).map((item, idx) => (
                       <div key={idx} className="flex justify-between items-center text-xs">
                         <span className={item.w === currentPair.target ? "font-bold text-green-700" : "text-gray-600"}>{item.w}</span>
                         <div className="flex-1 mx-2 bg-gray-200 rounded-full h-1.5 overflow-hidden">
                           <div className="bg-purple-500 h-full" style={{ width: `${item.p * 100}%` }} />
                         </div>
                         <span className="font-mono text-[10px]">{item.p.toFixed(2)}</span>
                       </div>
                     ))}
                   </div>
                 ) : <div className="text-center text-xs text-gray-400">No prediction yet</div>}
              </div>

              <div className="flex justify-center"><ArrowRight className="rotate-90 text-gray-300" size={20}/></div>

              {/* Step 3: Update */}
              <div className={`p-3 rounded border transition-all ${isTraining ? 'bg-amber-50 border-amber-300' : 'bg-gray-50 border-gray-200'}`}>
                 <div className="text-xs font-bold text-gray-500 mb-1">3. Weight Update (Backpropagation)</div>
                 <div className="text-xs text-gray-600 leading-snug">
                   Calculating gradient... <br/>
                   Updating weights for row <strong>"{currentPair?.input || '?'}"</strong> to favor <strong>"{currentPair?.target || '?'}"</strong>.
                 </div>
              </div>

            </div>
          </div>

          {/* Test Panel */}
          <div className="bg-white p-5 rounded-xl shadow-sm border border-gray-200">
             <h2 className="font-bold text-gray-800 mb-4 flex items-center gap-2">
               <Target size={18} /> Test Model Capabilities
             </h2>
             <div className="flex gap-2 mb-2">
               <select 
                 value={testInput} 
                 onChange={(e) => setTestInput(e.target.value)}
                 className="p-2 border rounded bg-gray-50 text-sm font-bold"
               >
                 {VOCAB.filter(w => w !== ".").map(w => <option key={w} value={w}>{w}</option>)}
               </select>
               <div className="flex items-center justify-center px-2">
                 <ArrowRight className="text-gray-400" />
               </div>
               <div className={`flex-1 p-2 rounded border font-mono text-center font-bold flex items-center justify-center gap-2 ${testOutput === (TOKEN_MAP[testInput] !== undefined ? TRAINING_DATA.find(d => d.input === testInput)?.target : "") ? "bg-green-100 text-green-800 border-green-300" : "bg-gray-100 text-gray-600"}`}>
                 {testOutput}
               </div>
             </div>
             <p className="text-xs text-gray-500">
               Initially, this is random. After ~50 steps, "Paris" should predict "is", "of" should predict "France", etc.
             </p>
          </div>

        </div>

        {/* RIGHT COLUMN: Weights Matrix */}
        <div className="space-y-6">
          <div className="bg-white p-5 rounded-xl shadow-sm border border-gray-200">
            <h2 className="font-bold text-gray-800 mb-2 flex items-center gap-2">
              <Brain size={18} /> Model Brain (Weights)
            </h2>
            <p className="text-xs text-gray-500 mb-4">
              This matrix represents the model's "Knowledge". 
              <br/>Row = Input, Col = Output likelihood.
              <br/><span className="text-green-600 font-bold">Green</span> = Strong connection (Learned).
            </p>
            
            {renderWeightMatrix()}

          </div>

          <ExplanationCard 
             title="How is 'Knowledge' Stored?"
             content="Notice how the weight matrix starts gray (random). As training continues, specific cells turn green. For example, the cell at Row 'Paris' / Col 'is' becomes green. This is physical 'knowledge'—a high value in a floating-point matrix."
          />
          
          <ExplanationCard 
             title="Interview Tip: Overfitting"
             content="If we train too long on this tiny dataset, the model will memorize it perfectly (100% probability) but might fail if we gave it a new sentence. In this simple demo, 'memorization' is exactly what we want, but in real life, we want 'generalization'."
          />
        </div>

      </div>
    </div>
  );
}