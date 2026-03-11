import { useEffect, useState } from 'react';
import { FlaskConical, ChevronRight, ChevronLeft, Play } from 'lucide-react';
import { fetchDatasets, fetchProfilerDefaults, runProfiler, fetchProfilerRuns, fetchProfilerRun } from '../api/client';
import { useJobPolling } from '../hooks/useWebSocket';
import StatusBadge from '../components/shared/StatusBadge';
import type { DatasetMeta, ProfilerRun } from '../types';

const ALL_MODELS = [
  { key: 'svc', label: 'SVC', cat: 'classical' },
  { key: 'dt', label: 'Decision Tree', cat: 'classical' },
  { key: 'lr', label: 'Logistic Regression', cat: 'classical' },
  { key: 'nb', label: 'Naive Bayes', cat: 'classical' },
  { key: 'rf', label: 'Random Forest', cat: 'classical' },
  { key: 'mlp', label: 'MLP', cat: 'classical' },
  { key: 'xgb', label: 'XGBoost', cat: 'classical' },
  { key: 'qsvc', label: 'Quantum SVC', cat: 'quantum' },
  { key: 'qnn', label: 'QNN', cat: 'quantum' },
  { key: 'vqc', label: 'VQC', cat: 'quantum' },
  { key: 'pqk', label: 'PQK', cat: 'quantum' },
];

const EMBEDDINGS = ['none', 'pca', 'nmf', 'lle', 'isomap', 'spectral', 'umap'];

export default function QProfiler() {
  const [step, setStep] = useState(0);
  const [datasets, setDatasets] = useState<DatasetMeta[]>([]);
  const [selectedDs, setSelectedDs] = useState<string[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>(['svc', 'dt', 'lr', 'nb', 'rf', 'mlp']);
  const [selectedEmb, setSelectedEmb] = useState<string[]>(['none', 'pca']);
  const [iterations, setIterations] = useState(2);
  const [testSize, setTestSize] = useState(0.3);
  const [nComponents, setNComponents] = useState(3);
  const [nJobs, setNJobs] = useState(4);
  const [jobId, setJobId] = useState<string | null>(null);
  const [runs, setRuns] = useState<ProfilerRun[]>([]);
  const [viewRun, setViewRun] = useState<any>(null);
  const jobState = useJobPolling(jobId);

  useEffect(() => {
    fetchDatasets().then(r => setDatasets(r.data.datasets || [])).catch(() => {});
    fetchProfilerRuns().then(r => setRuns(r.data.runs || [])).catch(() => {});
  }, []);

  const toggleDs = (id: string) => {
    setSelectedDs(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]);
  };
  const toggleModel = (key: string) => {
    setSelectedModels(prev => prev.includes(key) ? prev.filter(x => x !== key) : [...prev, key]);
  };
  const toggleEmb = (key: string) => {
    setSelectedEmb(prev => prev.includes(key) ? prev.filter(x => x !== key) : [...prev, key]);
  };

  const handleRun = async () => {
    try {
      const r = await runProfiler({
        dataset_ids: selectedDs,
        models: selectedModels,
        embeddings: selectedEmb,
        iterations, test_size: testSize, n_components: nComponents, n_jobs: nJobs,
      });
      setJobId(r.data.job_id);
      setStep(4);
    } catch (e: any) {
      alert(e?.response?.data?.detail || 'Profiler failed');
    }
  };

  const handleViewRun = async (runId: string) => {
    const r = await fetchProfilerRun(runId);
    setViewRun(r.data);
  };

  const steps = ['Datasets', 'Models', 'Embeddings', 'Config', 'Run'];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
          <FlaskConical className="w-6 h-6 text-purple-600" /> QProfiler
        </h1>
        <p className="text-slate-500 mt-1">Automated quantum vs. classical ML benchmarking pipeline</p>
      </div>

      {/* Step indicator */}
      <div className="flex items-center gap-2">
        {steps.map((s, i) => (
          <div key={s} className="flex items-center gap-2">
            <button onClick={() => setStep(i)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
                step === i ? 'bg-purple-600 text-white' : i < step ? 'bg-purple-100 text-purple-700' : 'bg-slate-100 text-slate-500'
              }`}>
              <span className="w-5 h-5 rounded-full bg-white/20 flex items-center justify-center text-xs">{i + 1}</span>
              {s}
            </button>
            {i < steps.length - 1 && <ChevronRight className="w-4 h-4 text-slate-300" />}
          </div>
        ))}
      </div>

      <div className="bg-white rounded-xl border border-slate-200 p-6">
        {/* Step 0: Datasets */}
        {step === 0 && (
          <div>
            <h3 className="font-semibold text-slate-900 mb-3">Select Datasets</h3>
            {datasets.length === 0 ? (
              <p className="text-slate-500">No datasets available. Upload or generate some first.</p>
            ) : (
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {datasets.map(d => (
                  <label key={d.id} className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all ${
                    selectedDs.includes(d.id) ? 'border-purple-400 bg-purple-50' : 'border-slate-200 hover:border-slate-300'
                  }`}>
                    <input type="checkbox" checked={selectedDs.includes(d.id)} onChange={() => toggleDs(d.id)}
                      className="rounded text-purple-600" />
                    <div>
                      <p className="text-sm font-medium text-slate-900">{d.name}</p>
                      <p className="text-xs text-slate-500">{d.rows}×{d.columns} &middot; {d.origin}</p>
                    </div>
                  </label>
                ))}
              </div>
            )}
            <p className="text-sm text-slate-500 mt-2">{selectedDs.length} dataset(s) selected</p>
          </div>
        )}

        {/* Step 1: Models */}
        {step === 1 && (
          <div>
            <h3 className="font-semibold text-slate-900 mb-3">Select Models</h3>
            <div className="space-y-4">
              <div>
                <p className="text-xs font-medium text-slate-500 uppercase mb-2">Classical</p>
                <div className="flex flex-wrap gap-2">
                  {ALL_MODELS.filter(m => m.cat === 'classical').map(m => (
                    <button key={m.key} onClick={() => toggleModel(m.key)}
                      className={`px-3 py-1.5 rounded-lg text-sm font-medium ${
                        selectedModels.includes(m.key) ? 'bg-cyan-600 text-white' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                      }`}>{m.label}</button>
                  ))}
                </div>
              </div>
              <div>
                <p className="text-xs font-medium text-slate-500 uppercase mb-2">Quantum</p>
                <div className="flex flex-wrap gap-2">
                  {ALL_MODELS.filter(m => m.cat === 'quantum').map(m => (
                    <button key={m.key} onClick={() => toggleModel(m.key)}
                      className={`px-3 py-1.5 rounded-lg text-sm font-medium ${
                        selectedModels.includes(m.key) ? 'bg-purple-600 text-white' : 'bg-purple-50 text-purple-700 hover:bg-purple-100'
                      }`}>{m.label}</button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Step 2: Embeddings */}
        {step === 2 && (
          <div>
            <h3 className="font-semibold text-slate-900 mb-3">Select Embedding Methods</h3>
            <div className="flex flex-wrap gap-2">
              {EMBEDDINGS.map(e => (
                <button key={e} onClick={() => toggleEmb(e)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium ${
                    selectedEmb.includes(e) ? 'bg-cyan-600 text-white' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                  }`}>{e === 'none' ? 'None (Raw)' : e.toUpperCase()}</button>
              ))}
            </div>
          </div>
        )}

        {/* Step 3: Config */}
        {step === 3 && (
          <div>
            <h3 className="font-semibold text-slate-900 mb-3">Pipeline Configuration</h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Iterations</label>
                <input type="number" value={iterations} onChange={e => setIterations(Number(e.target.value))} min={1} max={20}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm" />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Test Size</label>
                <input type="number" value={testSize} onChange={e => setTestSize(Number(e.target.value))} step={0.05} min={0.1} max={0.5}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm" />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Components</label>
                <input type="number" value={nComponents} onChange={e => setNComponents(Number(e.target.value))} min={1} max={50}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm" />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Parallel Jobs</label>
                <input type="number" value={nJobs} onChange={e => setNJobs(Number(e.target.value))} min={1} max={16}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm" />
              </div>
            </div>

            <div className="mt-4 p-4 bg-slate-50 rounded-lg">
              <h4 className="text-sm font-medium text-slate-700 mb-2">Run Summary</h4>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-sm">
                <div><span className="text-slate-500">Datasets:</span> <strong>{selectedDs.length}</strong></div>
                <div><span className="text-slate-500">Models:</span> <strong>{selectedModels.length}</strong></div>
                <div><span className="text-slate-500">Embeddings:</span> <strong>{selectedEmb.length}</strong></div>
                <div><span className="text-slate-500">Total combos:</span> <strong>{selectedDs.length * iterations * selectedEmb.length * selectedModels.length}</strong></div>
              </div>
            </div>
          </div>
        )}

        {/* Step 4: Run */}
        {step === 4 && (
          <div>
            <h3 className="font-semibold text-slate-900 mb-3">Pipeline Execution</h3>
            {jobState ? (
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <StatusBadge status={jobState.status} />
                  <span className="text-sm text-slate-700">{jobState.message}</span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-3">
                  <div className="bg-purple-500 h-3 rounded-full transition-all duration-300" style={{ width: `${jobState.progress * 100}%` }} />
                </div>
                <p className="text-sm text-slate-500">{Math.round(jobState.progress * 100)}% complete</p>
              </div>
            ) : (
              <p className="text-slate-500">Click "Run Profiler" to start the pipeline.</p>
            )}
          </div>
        )}

        {/* Navigation */}
        <div className="flex items-center justify-between mt-6 pt-4 border-t border-slate-200">
          <button onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0}
            className="flex items-center gap-1 px-4 py-2 text-sm font-medium text-slate-600 hover:text-slate-900 disabled:opacity-30">
            <ChevronLeft className="w-4 h-4" /> Previous
          </button>
          {step < 3 ? (
            <button onClick={() => setStep(step + 1)}
              className="flex items-center gap-1 px-4 py-2 bg-purple-600 text-white rounded-lg text-sm font-medium hover:bg-purple-700">
              Next <ChevronRight className="w-4 h-4" />
            </button>
          ) : step === 3 ? (
            <button onClick={handleRun} disabled={selectedDs.length === 0}
              className="flex items-center gap-2 px-5 py-2.5 bg-purple-600 text-white rounded-lg font-medium hover:bg-purple-700 disabled:opacity-50">
              <Play className="w-4 h-4" /> Run Profiler
            </button>
          ) : null}
        </div>
      </div>

      {/* Previous runs */}
      {runs.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <h2 className="text-lg font-semibold text-slate-900 mb-4">Previous Runs</h2>
          <div className="space-y-2">
            {runs.map(r => (
              <div key={r.run_id} className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                <div>
                  <p className="text-sm font-medium text-slate-900">Run {r.run_id}</p>
                  <p className="text-xs text-slate-500">{new Date(r.created_at).toLocaleString()} &middot; {r.n_model_results} results</p>
                </div>
                <button onClick={() => handleViewRun(r.run_id)} className="text-sm text-cyan-600 font-medium hover:text-cyan-700">View Results</button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* View run results */}
      {viewRun && (
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-slate-900">Run Results</h2>
            <button onClick={() => setViewRun(null)} className="text-sm text-slate-400">Close</button>
          </div>
          {viewRun.model_results?.length > 0 && (
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-200">
                    <th className="px-3 py-2 text-left font-medium text-slate-600">Dataset</th>
                    <th className="px-3 py-2 text-left font-medium text-slate-600">Model</th>
                    <th className="px-3 py-2 text-left font-medium text-slate-600">Embedding</th>
                    <th className="px-3 py-2 text-left font-medium text-slate-600">Accuracy</th>
                    <th className="px-3 py-2 text-left font-medium text-slate-600">F1</th>
                  </tr>
                </thead>
                <tbody>
                  {viewRun.model_results.map((r: any, i: number) => (
                    <tr key={i} className="border-b border-slate-100">
                      <td className="px-3 py-2 text-slate-900">{r.Dataset}</td>
                      <td className="px-3 py-2 text-slate-700">{r.model}</td>
                      <td className="px-3 py-2 text-slate-700">{r.embeddings}</td>
                      <td className="px-3 py-2 font-mono">{(r.accuracy * 100).toFixed(1)}%</td>
                      <td className="px-3 py-2 font-mono">{(r.f1_score * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
