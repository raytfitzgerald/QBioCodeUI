import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Database, Sparkles, Cpu, FlaskConical, ListChecks, ArrowRight } from 'lucide-react';
import { fetchDatasets, fetchJobs, fetchProfilerRuns } from '../api/client';
import StatusBadge from '../components/shared/StatusBadge';
import type { DatasetMeta, Job, ProfilerRun } from '../types';

export default function Dashboard() {
  const [datasets, setDatasets] = useState<DatasetMeta[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [runs, setRuns] = useState<ProfilerRun[]>([]);

  useEffect(() => {
    fetchDatasets().then(r => setDatasets(r.data.datasets || [])).catch(() => {});
    fetchJobs().then(r => setJobs(r.data.jobs || [])).catch(() => {});
    fetchProfilerRuns().then(r => setRuns(r.data.runs || [])).catch(() => {});
  }, []);

  const stats = [
    { label: 'Datasets', value: datasets.length, icon: Database, to: '/datasets', color: 'bg-blue-500' },
    { label: 'Active Jobs', value: jobs.filter(j => j.status === 'running').length, icon: ListChecks, to: '/jobs', color: 'bg-amber-500' },
    { label: 'Profiler Runs', value: runs.length, icon: FlaskConical, to: '/profiler', color: 'bg-purple-500' },
    { label: 'Completed', value: jobs.filter(j => j.status === 'completed').length, icon: Cpu, to: '/jobs', color: 'bg-green-500' },
  ];

  const quickActions = [
    { label: 'Upload Dataset', description: 'Import a CSV file', to: '/datasets', icon: Database },
    { label: 'Generate Data', description: 'Create synthetic datasets', to: '/generate', icon: Sparkles },
    { label: 'Train Model', description: 'Classical or quantum ML', to: '/train', icon: Cpu },
    { label: 'Run QProfiler', description: 'Full benchmarking pipeline', to: '/profiler', icon: FlaskConical },
  ];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Dashboard</h1>
        <p className="mt-1 text-slate-500">Quantum ML benchmarking for healthcare and life sciences</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map(s => (
          <Link key={s.label} to={s.to} className="bg-white rounded-xl border border-slate-200 p-5 hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-500">{s.label}</p>
                <p className="text-3xl font-bold text-slate-900 mt-1">{s.value}</p>
              </div>
              <div className={`${s.color} p-3 rounded-lg`}>
                <s.icon className="w-5 h-5 text-white" />
              </div>
            </div>
          </Link>
        ))}
      </div>

      <div>
        <h2 className="text-lg font-semibold text-slate-900 mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {quickActions.map(a => (
            <Link key={a.label} to={a.to} className="bg-white rounded-xl border border-slate-200 p-5 hover:border-cyan-400 hover:shadow-md transition-all group">
              <a.icon className="w-8 h-8 text-cyan-600 mb-3" />
              <h3 className="font-semibold text-slate-900">{a.label}</h3>
              <p className="text-sm text-slate-500 mt-1">{a.description}</p>
              <div className="flex items-center gap-1 mt-3 text-sm text-cyan-600 font-medium group-hover:gap-2 transition-all">
                Get started <ArrowRight className="w-4 h-4" />
              </div>
            </Link>
          ))}
        </div>
      </div>

      {jobs.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-slate-900">Recent Jobs</h2>
            <Link to="/jobs" className="text-sm text-cyan-600 hover:text-cyan-700">View all</Link>
          </div>
          <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
            <table className="min-w-full divide-y divide-slate-200">
              <thead className="bg-slate-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase">ID</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase">Type</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase">Status</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase">Progress</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {jobs.slice(0, 5).map(j => (
                  <tr key={j.id} className="hover:bg-slate-50">
                    <td className="px-4 py-3 text-sm font-mono text-slate-600">{j.id}</td>
                    <td className="px-4 py-3 text-sm text-slate-900">{j.type}</td>
                    <td className="px-4 py-3"><StatusBadge status={j.status} /></td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <div className="flex-1 bg-slate-200 rounded-full h-2">
                          <div className="bg-cyan-500 h-2 rounded-full transition-all" style={{ width: `${j.progress * 100}%` }} />
                        </div>
                        <span className="text-xs text-slate-500">{Math.round(j.progress * 100)}%</span>
                      </div>
                    </td>
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
