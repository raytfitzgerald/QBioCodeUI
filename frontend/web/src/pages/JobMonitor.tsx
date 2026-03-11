import { useEffect, useState, useCallback } from 'react';
import { ListChecks, RefreshCw, XCircle } from 'lucide-react';
import { fetchJobs, cancelJob } from '../api/client';
import StatusBadge from '../components/shared/StatusBadge';
import type { Job } from '../types';

export default function JobMonitor() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [filter, setFilter] = useState('all');

  const load = useCallback(async () => {
    try {
      const r = await fetchJobs(filter === 'all' ? undefined : filter);
      setJobs(r.data.jobs || []);
    } catch { /* empty */ }
  }, [filter]);

  useEffect(() => { load(); const t = setInterval(load, 3000); return () => clearInterval(t); }, [load]);

  const handleCancel = async (id: string) => {
    await cancelJob(id);
    load();
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
            <ListChecks className="w-6 h-6 text-amber-600" /> Job Monitor
          </h1>
          <p className="text-slate-500 mt-1">Track all background tasks</p>
        </div>
        <button onClick={load} className="p-2 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-100">
          <RefreshCw className="w-5 h-5" />
        </button>
      </div>

      <div className="flex gap-2">
        {['all', 'running', 'completed', 'failed'].map(f => (
          <button key={f} onClick={() => setFilter(f)}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium ${
              filter === f ? 'bg-slate-900 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}>{f.charAt(0).toUpperCase() + f.slice(1)}</button>
        ))}
      </div>

      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <table className="min-w-full divide-y divide-slate-200">
          <thead className="bg-slate-50">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase">ID</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase">Type</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase">Status</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase">Progress</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase">Message</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase">Created</th>
              <th className="px-4 py-3 text-right text-xs font-medium text-slate-500 uppercase">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {jobs.length === 0 ? (
              <tr><td colSpan={7} className="px-4 py-8 text-center text-slate-400">No jobs found</td></tr>
            ) : jobs.map(j => (
              <tr key={j.id} className="hover:bg-slate-50">
                <td className="px-4 py-3 text-sm font-mono text-slate-600">{j.id}</td>
                <td className="px-4 py-3 text-sm text-slate-900">{j.type.replace(/_/g, ' ')}</td>
                <td className="px-4 py-3"><StatusBadge status={j.status} /></td>
                <td className="px-4 py-3 w-40">
                  <div className="flex items-center gap-2">
                    <div className="flex-1 bg-slate-200 rounded-full h-2">
                      <div className={`h-2 rounded-full transition-all ${
                        j.status === 'failed' ? 'bg-red-500' : j.status === 'completed' ? 'bg-green-500' : 'bg-cyan-500'
                      }`} style={{ width: `${j.progress * 100}%` }} />
                    </div>
                    <span className="text-xs text-slate-500 w-8">{Math.round(j.progress * 100)}%</span>
                  </div>
                </td>
                <td className="px-4 py-3 text-sm text-slate-500 max-w-xs truncate">{j.message || j.error || '—'}</td>
                <td className="px-4 py-3 text-sm text-slate-500">{new Date(j.created_at).toLocaleTimeString()}</td>
                <td className="px-4 py-3 text-right">
                  {j.status === 'running' && (
                    <button onClick={() => handleCancel(j.id)} className="p-1.5 text-red-400 hover:text-red-600" title="Cancel">
                      <XCircle className="w-4 h-4" />
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
