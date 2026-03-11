import { useEffect, useState } from 'react';
import { BarChart3 } from 'lucide-react';
import { fetchDatasets, evaluateDataset, fetchMetricsInfo } from '../api/client';
import { useJobPolling } from '../hooks/useWebSocket';
import StatusBadge from '../components/shared/StatusBadge';
import type { DatasetMeta, EvaluationMetric } from '../types';

export default function ComplexityEvaluation() {
  const [datasets, setDatasets] = useState<DatasetMeta[]>([]);
  const [metricsInfo, setMetricsInfo] = useState<EvaluationMetric[]>([]);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [labelColumn, setLabelColumn] = useState('class');
  const [results, setResults] = useState<Record<string, any> | null>(null);
  const [loading, setLoading] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const jobState = useJobPolling(jobId);

  useEffect(() => {
    fetchDatasets().then(r => setDatasets(r.data.datasets || [])).catch(() => {});
    fetchMetricsInfo().then(r => setMetricsInfo(r.data.metrics || [])).catch(() => {});
  }, []);

  useEffect(() => {
    const ds = datasets.find(d => d.id === selectedDataset);
    if (ds?.label_column) setLabelColumn(ds.label_column);
  }, [selectedDataset, datasets]);

  const handleEvaluate = async () => {
    if (!selectedDataset) return;
    setLoading(true);
    setResults(null);
    try {
      const r = await evaluateDataset(selectedDataset, labelColumn);
      if (r.data.job_id) {
        setJobId(r.data.job_id);
      } else {
        setResults(r.data.metrics || r.data);
      }
    } catch (e: any) {
      alert(e?.response?.data?.detail || 'Evaluation failed');
    }
    setLoading(false);
  };

  useEffect(() => {
    if (jobState?.status === 'completed' && jobState.result) {
      setResults((jobState.result as any).metrics || jobState.result);
    }
  }, [jobState]);

  const selectedDs = datasets.find(d => d.id === selectedDataset);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Complexity Evaluation</h1>
        <p className="text-slate-500 mt-1">Compute 22+ dataset complexity metrics to understand data structure</p>
      </div>

      <div className="bg-white rounded-xl border border-slate-200 p-6">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Dataset</label>
            <select
              value={selectedDataset}
              onChange={e => setSelectedDataset(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-cyan-500"
            >
              <option value="">Select a dataset...</option>
              {datasets.map(d => (
                <option key={d.id} value={d.id}>{d.name} ({d.rows}×{d.columns})</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Label Column</label>
            <select
              value={labelColumn}
              onChange={e => setLabelColumn(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-cyan-500"
            >
              {selectedDs?.column_names.map(c => (
                <option key={c} value={c}>{c}</option>
              )) || <option value="class">class</option>}
            </select>
          </div>
          <div className="flex items-end">
            <button
              onClick={handleEvaluate}
              disabled={!selectedDataset || loading}
              className="w-full px-4 py-2 bg-cyan-600 text-white rounded-lg font-medium hover:bg-cyan-700 disabled:opacity-50"
            >
              {loading ? 'Evaluating...' : 'Evaluate'}
            </button>
          </div>
        </div>
        {jobState && <div className="mt-3"><StatusBadge status={jobState.status} /> <span className="text-sm text-slate-500 ml-2">{jobState.message}</span></div>}
      </div>

      {results && (
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <h2 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-cyan-600" /> Complexity Metrics
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {metricsInfo.map(m => {
              const val = results[m.key];
              if (val === undefined || val === null) return null;
              return (
                <div key={m.key} className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                  <p className="text-xs text-slate-500">{m.label}</p>
                  <p className="text-lg font-semibold text-slate-900 font-mono mt-0.5">
                    {typeof val === 'number' ? (Number.isInteger(val) ? val : val.toFixed(4)) : String(val)}
                  </p>
                  <p className="text-xs text-slate-400 mt-0.5">{m.description}</p>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
