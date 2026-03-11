import { useEffect, useState } from 'react';
import { BrainCircuit } from 'lucide-react';
import { fetchProfilerRuns, fetchDatasets, trainSage, predictSage, fetchSageModels } from '../api/client';
import { useJobPolling } from '../hooks/useWebSocket';
import StatusBadge from '../components/shared/StatusBadge';
import type { ProfilerRun, DatasetMeta, SageModel } from '../types';

export default function QSage() {
  const [tab, setTab] = useState<'train' | 'predict'>('train');
  const [runs, setRuns] = useState<ProfilerRun[]>([]);
  const [datasets, setDatasets] = useState<DatasetMeta[]>([]);
  const [sageModels, setSageModels] = useState<SageModel[]>([]);
  const [selectedRun, setSelectedRun] = useState('');
  const [sageType, setSageType] = useState('random_forest');
  const [selectedSage, setSelectedSage] = useState('');
  const [selectedDs, setSelectedDs] = useState('');
  const [metric, setMetric] = useState('f1_score');
  const [jobId, setJobId] = useState<string | null>(null);
  const jobState = useJobPolling(jobId);

  useEffect(() => {
    fetchProfilerRuns().then(r => setRuns(r.data.runs || [])).catch(() => {});
    fetchDatasets().then(r => setDatasets(r.data.datasets || [])).catch(() => {});
    fetchSageModels().then(r => setSageModels(r.data.models || [])).catch(() => {});
  }, []);

  const handleTrain = async () => {
    if (!selectedRun) return;
    try {
      const r = await trainSage({ profiler_run_id: selectedRun, sage_type: sageType });
      setJobId(r.data.job_id);
    } catch (e: any) {
      alert(e?.response?.data?.detail || 'Training failed');
    }
  };

  const handlePredict = async () => {
    if (!selectedSage || !selectedDs) return;
    try {
      const r = await predictSage({ sage_model_id: selectedSage, dataset_id: selectedDs, metric });
      setJobId(r.data.job_id);
    } catch (e: any) {
      alert(e?.response?.data?.detail || 'Prediction failed');
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
          <BrainCircuit className="w-6 h-6 text-emerald-600" /> QSage
        </h1>
        <p className="text-slate-500 mt-1">Meta-learning model selector — predict the best model for your dataset</p>
      </div>

      <div className="flex gap-2 border-b border-slate-200">
        <button onClick={() => setTab('train')}
          className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
            tab === 'train' ? 'border-emerald-600 text-emerald-700' : 'border-transparent text-slate-500 hover:text-slate-700'
          }`}>Train QSage</button>
        <button onClick={() => setTab('predict')}
          className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
            tab === 'predict' ? 'border-emerald-600 text-emerald-700' : 'border-transparent text-slate-500 hover:text-slate-700'
          }`}>Predict Best Model</button>
      </div>

      {tab === 'train' && (
        <div className="bg-white rounded-xl border border-slate-200 p-6 space-y-4">
          <h3 className="font-semibold text-slate-900">Train on Profiler Results</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Profiler Run</label>
              <select value={selectedRun} onChange={e => setSelectedRun(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-emerald-500">
                <option value="">Select a run...</option>
                {runs.map(r => (
                  <option key={r.run_id} value={r.run_id}>Run {r.run_id} ({r.n_model_results} results)</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Sage Type</label>
              <select value={sageType} onChange={e => setSageType(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-emerald-500">
                <option value="random_forest">Random Forest</option>
                <option value="mlp">Multi-Layer Perceptron</option>
              </select>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <button onClick={handleTrain} disabled={!selectedRun || (!!jobId && jobState?.status === 'running')}
              className="px-5 py-2.5 bg-emerald-600 text-white rounded-lg font-medium hover:bg-emerald-700 disabled:opacity-50">
              {jobId && jobState?.status === 'running' ? 'Training...' : 'Train QSage'}
            </button>
            {jobState && <StatusBadge status={jobState.status} />}
            {jobState && <span className="text-sm text-slate-500">{jobState.message}</span>}
          </div>

          {sageModels.length > 0 && (
            <div className="mt-4">
              <h4 className="text-sm font-medium text-slate-700 mb-2">Trained Models</h4>
              <div className="space-y-2">
                {sageModels.map(m => (
                  <div key={m.sage_id} className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                    <div>
                      <p className="text-sm font-medium">{m.sage_type} — {m.sage_id}</p>
                      <p className="text-xs text-slate-500">{new Date(m.created_at).toLocaleString()}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {tab === 'predict' && (
        <div className="bg-white rounded-xl border border-slate-200 p-6 space-y-4">
          <h3 className="font-semibold text-slate-900">Predict Best Model for New Dataset</h3>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">QSage Model</label>
              <select value={selectedSage} onChange={e => setSelectedSage(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm">
                <option value="">Select...</option>
                {sageModels.map(m => <option key={m.sage_id} value={m.sage_id}>{m.sage_type} — {m.sage_id}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Dataset</label>
              <select value={selectedDs} onChange={e => setSelectedDs(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm">
                <option value="">Select...</option>
                {datasets.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Metric</label>
              <select value={metric} onChange={e => setMetric(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm">
                <option value="f1_score">F1 Score</option>
                <option value="accuracy">Accuracy</option>
                <option value="auc">AUC</option>
              </select>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <button onClick={handlePredict} disabled={!selectedSage || !selectedDs}
              className="px-5 py-2.5 bg-emerald-600 text-white rounded-lg font-medium hover:bg-emerald-700 disabled:opacity-50">
              Predict
            </button>
            {jobState && <StatusBadge status={jobState.status} />}
          </div>
          {jobState?.status === 'completed' && jobState.result && (
            <div className="mt-4 p-4 bg-emerald-50 rounded-lg">
              <h4 className="font-medium text-emerald-900 mb-2">Predictions</h4>
              <pre className="text-sm text-slate-700 overflow-x-auto">{JSON.stringify(jobState.result, null, 2)}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
