import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Layers } from 'lucide-react';
import { fetchDatasets, fetchEmbeddingMethods, computeEmbedding } from '../api/client';
import { useJobPolling } from '../hooks/useWebSocket';
import StatusBadge from '../components/shared/StatusBadge';
import type { DatasetMeta, EmbeddingMethod } from '../types';

export default function Embeddings() {
  const navigate = useNavigate();
  const [datasets, setDatasets] = useState<DatasetMeta[]>([]);
  const [methods, setMethods] = useState<EmbeddingMethod[]>([]);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [selectedMethod, setSelectedMethod] = useState('pca');
  const [labelColumn, setLabelColumn] = useState('class');
  const [nComponents, setNComponents] = useState(3);
  const [nNeighbors, setNNeighbors] = useState(30);
  const [jobId, setJobId] = useState<string | null>(null);
  const jobState = useJobPolling(jobId);

  useEffect(() => {
    fetchDatasets().then(r => setDatasets(r.data.datasets || [])).catch(() => {});
    fetchEmbeddingMethods().then(r => setMethods(r.data.methods || [])).catch(() => {});
  }, []);

  useEffect(() => {
    const ds = datasets.find(d => d.id === selectedDataset);
    if (ds?.label_column) setLabelColumn(ds.label_column);
  }, [selectedDataset, datasets]);

  const handleCompute = async () => {
    if (!selectedDataset || !selectedMethod) return;
    try {
      const r = await computeEmbedding({
        dataset_id: selectedDataset,
        method: selectedMethod,
        label_column: labelColumn,
        n_components: nComponents,
        n_neighbors: nNeighbors,
      });
      setJobId(r.data.job_id);
    } catch (e: any) {
      alert(e?.response?.data?.detail || 'Embedding failed');
    }
  };

  const methodInfo = methods.find(m => m.method === selectedMethod);
  const needsNeighbors = methodInfo?.params.includes('n_neighbors');
  const selectedDs = datasets.find(d => d.id === selectedDataset);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Dimensionality Reduction</h1>
        <p className="text-slate-500 mt-1">Embed datasets using classical or quantum methods</p>
      </div>

      <div className="bg-white rounded-xl border border-slate-200 p-6 space-y-5">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Dataset</label>
            <select value={selectedDataset} onChange={e => setSelectedDataset(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-cyan-500">
              <option value="">Select...</option>
              {datasets.map(d => <option key={d.id} value={d.id}>{d.name} ({d.rows}×{d.columns})</option>)}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Label Column</label>
            <select value={labelColumn} onChange={e => setLabelColumn(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-cyan-500">
              {selectedDs?.column_names.map(c => <option key={c} value={c}>{c}</option>) || <option value="class">class</option>}
            </select>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">Embedding Method</label>
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2">
            {methods.map(m => (
              <button key={m.method} onClick={() => setSelectedMethod(m.method)}
                className={`p-3 rounded-lg border-2 text-center text-sm transition-all ${
                  selectedMethod === m.method ? 'border-cyan-500 bg-cyan-50' : 'border-slate-200 hover:border-slate-300 bg-white'
                }`}>
                <Layers className={`w-5 h-5 mx-auto mb-1 ${selectedMethod === m.method ? 'text-cyan-600' : 'text-slate-400'}`} />
                {m.label}
              </button>
            ))}
          </div>
          {methodInfo && <p className="text-sm text-slate-500 mt-2">{methodInfo.description}</p>}
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Components</label>
            <input type="number" value={nComponents} onChange={e => setNComponents(Number(e.target.value))} min={1} max={50}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-cyan-500" />
          </div>
          {needsNeighbors && (
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Neighbors</label>
              <input type="number" value={nNeighbors} onChange={e => setNNeighbors(Number(e.target.value))} min={2} max={200}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-cyan-500" />
            </div>
          )}
        </div>

        <div className="flex items-center gap-4">
          <button onClick={handleCompute} disabled={!selectedDataset || (!!jobId && jobState?.status === 'running')}
            className="px-5 py-2.5 bg-cyan-600 text-white rounded-lg font-medium hover:bg-cyan-700 disabled:opacity-50">
            {jobId && jobState?.status === 'running' ? 'Computing...' : 'Compute Embedding'}
          </button>
          {jobState && (
            <div className="flex items-center gap-3">
              <StatusBadge status={jobState.status} />
              <span className="text-sm text-slate-500">{jobState.message}</span>
              {jobState.status === 'completed' && (
                <button onClick={() => navigate('/datasets')} className="text-sm text-cyan-600 font-medium">View Result</button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
