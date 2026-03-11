import { useEffect, useState } from 'react';
import { Cpu, Atom } from 'lucide-react';
import { fetchDatasets, fetchAvailableModels, trainModel } from '../api/client';
import { useJobPolling } from '../hooks/useWebSocket';
import StatusBadge from '../components/shared/StatusBadge';
import type { DatasetMeta, ModelSchema } from '../types';

export default function ModelTraining() {
  const [datasets, setDatasets] = useState<DatasetMeta[]>([]);
  const [models, setModels] = useState<Record<string, ModelSchema>>({});
  const [selectedDataset, setSelectedDataset] = useState('');
  const [selectedModel, setSelectedModel] = useState('rf');
  const [labelColumn, setLabelColumn] = useState('class');
  const [testSize, setTestSize] = useState(0.3);
  const [scaling, setScaling] = useState('MinMaxScaler');
  const [gridSearch, setGridSearch] = useState(false);
  const [params, setParams] = useState<Record<string, string>>({});
  const [jobId, setJobId] = useState<string | null>(null);
  const [results, setResults] = useState<any[]>([]);
  const jobState = useJobPolling(jobId);

  useEffect(() => {
    fetchDatasets().then(r => setDatasets(r.data.datasets || [])).catch(() => {});
    fetchAvailableModels().then(r => setModels(r.data.models || {})).catch(() => {});
  }, []);

  useEffect(() => {
    const ds = datasets.find(d => d.id === selectedDataset);
    if (ds?.label_column) setLabelColumn(ds.label_column);
  }, [selectedDataset, datasets]);

  useEffect(() => {
    // Reset params when model changes
    const schema = models[selectedModel];
    if (schema) {
      const defaults: Record<string, string> = {};
      Object.entries(schema.params).forEach(([k, v]) => {
        defaults[k] = String(v.default ?? '');
      });
      setParams(defaults);
    }
  }, [selectedModel, models]);

  useEffect(() => {
    if (jobState?.status === 'completed' && jobState.result) {
      setResults(prev => [...prev, jobState.result as any]);
    }
  }, [jobState]);

  const handleTrain = async () => {
    if (!selectedDataset || !selectedModel) return;
    const parsedParams: Record<string, any> = {};
    const schema = models[selectedModel];
    if (schema) {
      Object.entries(params).forEach(([k, v]) => {
        const info = schema.params[k];
        if (!info || v === '' || v === 'null' || v === 'None') return;
        if (info.type === 'int') parsedParams[k] = parseInt(v);
        else if (info.type === 'float') parsedParams[k] = parseFloat(v);
        else if (info.type === 'bool') parsedParams[k] = v === 'true';
        else parsedParams[k] = v;
      });
    }
    try {
      const r = await trainModel({
        dataset_id: selectedDataset,
        model: selectedModel,
        label_column: labelColumn,
        test_size: testSize,
        scaling,
        grid_search: gridSearch,
        params: parsedParams,
      });
      setJobId(r.data.job_id);
    } catch (e: any) {
      alert(e?.response?.data?.detail || 'Training failed');
    }
  };

  const schema = models[selectedModel];
  const classicalModels = Object.entries(models).filter(([, v]) => v.category === 'classical');
  const quantumModels = Object.entries(models).filter(([, v]) => v.category === 'quantum');
  const selectedDs = datasets.find(d => d.id === selectedDataset);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Model Training</h1>
        <p className="text-slate-500 mt-1">Train classical and quantum ML models on your datasets</p>
      </div>

      <div className="bg-white rounded-xl border border-slate-200 p-6 space-y-5">
        {/* Dataset & config */}
        <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Dataset</label>
            <select value={selectedDataset} onChange={e => setSelectedDataset(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-cyan-500">
              <option value="">Select...</option>
              {datasets.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Label Column</label>
            <select value={labelColumn} onChange={e => setLabelColumn(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-cyan-500">
              {selectedDs?.column_names.map(c => <option key={c} value={c}>{c}</option>) || <option value="class">class</option>}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Test Size</label>
            <input type="number" value={testSize} step={0.05} min={0.1} max={0.5}
              onChange={e => setTestSize(Number(e.target.value))}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-cyan-500" />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Scaling</label>
            <select value={scaling} onChange={e => setScaling(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-cyan-500">
              <option value="MinMaxScaler">MinMaxScaler</option>
              <option value="StandardScaler">StandardScaler</option>
              <option value="None">None</option>
            </select>
          </div>
        </div>

        {/* Model selector */}
        <div>
          <h3 className="text-sm font-medium text-slate-700 mb-2 flex items-center gap-2">
            <Cpu className="w-4 h-4" /> Classical Models
          </h3>
          <div className="flex flex-wrap gap-2 mb-3">
            {classicalModels.map(([key, m]) => (
              <button key={key} onClick={() => setSelectedModel(key)}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                  selectedModel === key ? 'bg-cyan-600 text-white' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                }`}>{m.label}</button>
            ))}
          </div>
          <h3 className="text-sm font-medium text-slate-700 mb-2 flex items-center gap-2">
            <Atom className="w-4 h-4" /> Quantum Models
          </h3>
          <div className="flex flex-wrap gap-2">
            {quantumModels.map(([key, m]) => (
              <button key={key} onClick={() => setSelectedModel(key)}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                  selectedModel === key ? 'bg-purple-600 text-white' : 'bg-purple-50 text-purple-700 hover:bg-purple-100'
                }`}>{m.label}</button>
            ))}
          </div>
        </div>

        {/* Model params */}
        {schema && (
          <div>
            <h3 className="text-sm font-medium text-slate-700 mb-3">{schema.label} Parameters</h3>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {Object.entries(schema.params).map(([key, info]) => (
                <div key={key}>
                  <label className="block text-xs font-medium text-slate-600 mb-1">{key}</label>
                  {info.type === 'select' ? (
                    <select value={params[key] || ''} onChange={e => setParams({ ...params, [key]: e.target.value })}
                      className="w-full px-3 py-1.5 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-cyan-500">
                      {info.options?.map((o: string) => <option key={o} value={o}>{o}</option>)}
                    </select>
                  ) : info.type === 'bool' ? (
                    <select value={params[key] || 'false'} onChange={e => setParams({ ...params, [key]: e.target.value })}
                      className="w-full px-3 py-1.5 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-cyan-500">
                      <option value="true">true</option>
                      <option value="false">false</option>
                    </select>
                  ) : (
                    <input type="text" value={params[key] || ''} onChange={e => setParams({ ...params, [key]: e.target.value })}
                      className="w-full px-3 py-1.5 border border-slate-300 rounded-lg text-sm font-mono focus:ring-2 focus:ring-cyan-500" />
                  )}
                  {info.description && <p className="text-xs text-slate-400 mt-0.5">{info.description}</p>}
                </div>
              ))}
            </div>
            <label className="flex items-center gap-2 mt-3">
              <input type="checkbox" checked={gridSearch} onChange={e => setGridSearch(e.target.checked)}
                className="rounded text-cyan-600 focus:ring-cyan-500" />
              <span className="text-sm text-slate-700">Enable grid search optimization</span>
            </label>
          </div>
        )}

        <div className="flex items-center gap-4">
          <button onClick={handleTrain} disabled={!selectedDataset || (!!jobId && jobState?.status === 'running')}
            className="px-5 py-2.5 bg-cyan-600 text-white rounded-lg font-medium hover:bg-cyan-700 disabled:opacity-50">
            {jobId && jobState?.status === 'running' ? 'Training...' : 'Train Model'}
          </button>
          {jobState && (
            <div className="flex items-center gap-3">
              <StatusBadge status={jobState.status} />
              <span className="text-sm text-slate-500">{jobState.message}</span>
            </div>
          )}
        </div>
      </div>

      {/* Results */}
      {results.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <h2 className="text-lg font-semibold text-slate-900 mb-4">Training Results</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200">
                  <th className="px-3 py-2 text-left font-medium text-slate-600">Model</th>
                  <th className="px-3 py-2 text-left font-medium text-slate-600">Accuracy</th>
                  <th className="px-3 py-2 text-left font-medium text-slate-600">F1 Score</th>
                  <th className="px-3 py-2 text-left font-medium text-slate-600">AUC</th>
                  <th className="px-3 py-2 text-left font-medium text-slate-600">Time (s)</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r, i) => (
                  <tr key={i} className="border-b border-slate-100">
                    <td className="px-3 py-2 font-medium text-slate-900">{r.model}</td>
                    <td className="px-3 py-2 text-slate-700 font-mono">{(r.accuracy * 100).toFixed(1)}%</td>
                    <td className="px-3 py-2 text-slate-700 font-mono">{(r.f1_score * 100).toFixed(1)}%</td>
                    <td className="px-3 py-2 text-slate-700 font-mono">{r.auc != null ? (r.auc * 100).toFixed(1) + '%' : '—'}</td>
                    <td className="px-3 py-2 text-slate-700 font-mono">{r.time?.toFixed(2) || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
