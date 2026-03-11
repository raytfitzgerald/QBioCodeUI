import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Sparkles, Circle, Moon, Waves, Globe, Shell, Dna } from 'lucide-react';
import { fetchGeneratorTypes, generateData } from '../api/client';
import { useJobPolling } from '../hooks/useWebSocket';
import StatusBadge from '../components/shared/StatusBadge';

const ICONS: Record<string, any> = {
  circles: Circle, moons: Moon, classes: Sparkles, s_curve: Waves,
  spheres: Globe, spirals: Dna, swiss_roll: Shell,
};

const PARAM_LABELS: Record<string, string> = {
  n_samples: 'Sample Sizes', noise: 'Noise Levels', hole: 'Hole Variants',
  n_classes: 'Number of Classes', dim: 'Dimensions', rad: 'Radii',
  n_features: 'Feature Counts', n_informative: 'Informative Features',
  n_redundant: 'Redundant Features', n_clusters_per_class: 'Clusters/Class',
  weights: 'Class Weights',
};

export default function DataGeneration() {
  const navigate = useNavigate();
  const [types, setTypes] = useState<any[]>([]);
  const [selectedType, setSelectedType] = useState('');
  const [params, setParams] = useState<Record<string, string>>({});
  const [saveName, setSaveName] = useState('generated');
  const [jobId, setJobId] = useState<string | null>(null);
  const jobState = useJobPolling(jobId);

  useEffect(() => {
    fetchGeneratorTypes().then(r => setTypes(r.data.types || [])).catch(() => {});
  }, []);

  const selectedInfo = types.find(t => t.type === selectedType);

  const handleGenerate = async () => {
    if (!selectedType) return;
    const parsedParams: Record<string, any> = { type: selectedType, save_name: saveName };
    if (selectedInfo) {
      for (const p of selectedInfo.params) {
        const val = params[p];
        if (val) {
          try {
            parsedParams[p] = JSON.parse(val);
          } catch {
            parsedParams[p] = val.split(',').map((v: string) => {
              const n = Number(v.trim());
              return isNaN(n) ? v.trim() : n;
            });
          }
        }
      }
    }
    try {
      const r = await generateData(parsedParams);
      setJobId(r.data.job_id);
    } catch (e: any) {
      alert(e?.response?.data?.detail || 'Generation failed');
    }
  };

  const DEFAULTS: Record<string, Record<string, string>> = {
    circles: { n_samples: '[100, 200]', noise: '[0.1, 0.3]' },
    moons: { n_samples: '[100, 200]', noise: '[0.1, 0.3]' },
    s_curve: { n_samples: '[100, 200]', noise: '[0.1, 0.3]' },
    swiss_roll: { n_samples: '[100, 200]', noise: '[0.1, 0.3]', hole: '[true, false]' },
    spheres: { n_samples: '[100, 200]', dim: '[3, 6]', rad: '[3, 6]' },
    spirals: { n_samples: '[100, 200]', n_classes: '[2]', noise: '[0.1, 0.3]', dim: '[3, 6]' },
    classes: { n_samples: '[100, 200]', n_features: '[10, 30]', n_informative: '[2, 6]', n_redundant: '[2]', n_clusters_per_class: '[1]' },
  };

  useEffect(() => {
    if (selectedType && DEFAULTS[selectedType]) {
      setParams(DEFAULTS[selectedType]);
    }
  }, [selectedType]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Generate Synthetic Data</h1>
        <p className="text-slate-500 mt-1">Create datasets with controlled complexity for benchmarking</p>
      </div>

      {/* Type selector */}
      <div>
        <h2 className="text-sm font-medium text-slate-700 mb-3">Select Dataset Type</h2>
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
          {types.map(t => {
            const Icon = ICONS[t.type] || Sparkles;
            return (
              <button
                key={t.type}
                onClick={() => setSelectedType(t.type)}
                className={`p-4 rounded-xl border-2 text-center transition-all ${
                  selectedType === t.type
                    ? 'border-cyan-500 bg-cyan-50 shadow-sm'
                    : 'border-slate-200 hover:border-slate-300 bg-white'
                }`}
              >
                <Icon className={`w-8 h-8 mx-auto mb-2 ${selectedType === t.type ? 'text-cyan-600' : 'text-slate-400'}`} />
                <p className="text-sm font-medium text-slate-900">{t.label}</p>
              </button>
            );
          })}
        </div>
      </div>

      {/* Parameters */}
      {selectedInfo && (
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <h2 className="text-lg font-semibold text-slate-900 mb-1">{selectedInfo.label} Parameters</h2>
          <p className="text-sm text-slate-500 mb-4">{selectedInfo.description}</p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Batch Name</label>
              <input
                type="text"
                value={saveName}
                onChange={e => setSaveName(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
              />
            </div>
            {selectedInfo.params.map((p: string) => (
              <div key={p}>
                <label className="block text-sm font-medium text-slate-700 mb-1">{PARAM_LABELS[p] || p}</label>
                <input
                  type="text"
                  value={params[p] || ''}
                  onChange={e => setParams({ ...params, [p]: e.target.value })}
                  placeholder={`e.g. [100, 200]`}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm font-mono focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
                />
                <p className="text-xs text-slate-400 mt-0.5">JSON array of values</p>
              </div>
            ))}
          </div>

          <div className="mt-6 flex items-center gap-4">
            <button
              onClick={handleGenerate}
              disabled={!!jobId && jobState?.status === 'running'}
              className="px-5 py-2.5 bg-cyan-600 text-white rounded-lg font-medium hover:bg-cyan-700 disabled:opacity-50 transition-colors"
            >
              {jobId && jobState?.status === 'running' ? 'Generating...' : 'Generate Datasets'}
            </button>
            {jobState && (
              <div className="flex items-center gap-3">
                <StatusBadge status={jobState.status} />
                <span className="text-sm text-slate-500">{jobState.message}</span>
                {jobState.status === 'completed' && (
                  <button onClick={() => navigate('/datasets')} className="text-sm text-cyan-600 hover:text-cyan-700 font-medium">
                    View Datasets
                  </button>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
