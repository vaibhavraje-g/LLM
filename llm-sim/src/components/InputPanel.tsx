import React from 'react';
import { ModelConfig } from '../types';

interface InputPanelProps {
  input: string;
  setInput: (val: string) => void;
  config: ModelConfig;
  setConfig: (val: ModelConfig) => void;
  onRun: () => void;
  onGenerateNext: () => void;
  onAutoGenerate: () => void;
  isGenerating: boolean;
}

export const InputPanel: React.FC<InputPanelProps> = ({
  input,
  setInput,
  config,
  setConfig,
  onRun,
  onGenerateNext,
  onAutoGenerate,
  isGenerating
}) => {
  return (
    <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
      <h2 className="text-xl font-bold mb-4 text-slate-800">Input & Parameters</h2>
      
      <div className="mb-6">
        <label className="block text-sm font-medium text-slate-700 mb-2">Input Prompt</label>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 min-h-[100px]"
          placeholder="Type something like 'I like cats'..."
        />
      </div>

      <div className="space-y-4 mb-6">
        <div>
          <label className="flex justify-between text-sm font-medium text-slate-700 mb-1">
            <span>Temperature</span>
            <span className="text-slate-500">{config.temperature}</span>
          </label>
          <input
            type="range"
            min="0.1"
            max="2.0"
            step="0.1"
            value={config.temperature}
            onChange={(e) => setConfig({ ...config, temperature: parseFloat(e.target.value) })}
            className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer"
          />
        </div>

        <div>
          <label className="flex justify-between text-sm font-medium text-slate-700 mb-1">
            <span>Max Tokens</span>
            <span className="text-slate-500">{config.maxTokens}</span>
          </label>
          <input
            type="range"
            min="1"
            max="20"
            step="1"
            value={config.maxTokens}
            onChange={(e) => setConfig({ ...config, maxTokens: parseInt(e.target.value) })}
            className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer"
          />
        </div>
      </div>

      <div className="flex flex-col gap-2">
        <button
          onClick={onRun}
          disabled={isGenerating}
          className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Run Full Pipeline
        </button>
        <div className="flex gap-2">
          <button
            onClick={onGenerateNext}
            disabled={isGenerating}
            className="flex-1 py-2 px-4 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Step Next
          </button>
          <button
            onClick={onAutoGenerate}
            disabled={isGenerating}
            className="flex-1 py-2 px-4 bg-purple-600 hover:bg-purple-700 text-white font-semibold rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Auto Run
          </button>
        </div>
      </div>
    </div>
  );
};
