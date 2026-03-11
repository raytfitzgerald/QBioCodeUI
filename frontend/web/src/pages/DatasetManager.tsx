import { useEffect, useState, useCallback } from 'react';
import { Upload, Trash2, Download, Eye, Database } from 'lucide-react';
import { fetchDatasets, uploadDataset, deleteDataset as deleteDs, fetchDataset } from '../api/client';
import type { DatasetMeta, DatasetPreview } from '../types';

export default function DatasetManager() {
  const [datasets, setDatasets] = useState<DatasetMeta[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [selected, setSelected] = useState<DatasetPreview | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const r = await fetchDatasets();
      setDatasets(r.data.datasets || []);
    } catch { /* empty */ }
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleUpload = async (file: File) => {
    setUploading(true);
    try {
      await uploadDataset(file);
      await load();
    } catch (e: any) {
      alert(e?.response?.data?.detail || 'Upload failed');
    }
    setUploading(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file?.name.endsWith('.csv')) handleUpload(file);
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this dataset?')) return;
    await deleteDs(id);
    if (selected?.meta.id === id) setSelected(null);
    await load();
  };

  const handleView = async (id: string) => {
    const r = await fetchDataset(id);
    setSelected(r.data);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Datasets</h1>
          <p className="text-slate-500 mt-1">Upload and manage your CSV datasets</p>
        </div>
      </div>

      {/* Upload area */}
      <div
        className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
          dragOver ? 'border-cyan-400 bg-cyan-50' : 'border-slate-300 hover:border-slate-400'
        }`}
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        <Upload className="w-10 h-10 text-slate-400 mx-auto mb-3" />
        <p className="text-slate-600 font-medium">
          {uploading ? 'Uploading...' : 'Drag & drop a CSV file here'}
        </p>
        <p className="text-sm text-slate-400 mt-1">or</p>
        <label className="inline-block mt-2 px-4 py-2 bg-cyan-600 text-white rounded-lg cursor-pointer hover:bg-cyan-700 transition-colors text-sm font-medium">
          Browse Files
          <input type="file" accept=".csv" className="hidden" onChange={e => {
            const f = e.target.files?.[0];
            if (f) handleUpload(f);
          }} />
        </label>
      </div>

      {/* Dataset table */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <table className="min-w-full divide-y divide-slate-200">
          <thead className="bg-slate-50">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase">Name</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase">Origin</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase">Shape</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase">Label Column</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase">Created</th>
              <th className="px-4 py-3 text-right text-xs font-medium text-slate-500 uppercase">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {loading ? (
              <tr><td colSpan={6} className="px-4 py-8 text-center text-slate-400">Loading...</td></tr>
            ) : datasets.length === 0 ? (
              <tr><td colSpan={6} className="px-4 py-8 text-center text-slate-400">
                <Database className="w-8 h-8 mx-auto mb-2 text-slate-300" />
                No datasets yet. Upload a CSV to get started.
              </td></tr>
            ) : datasets.map(d => (
              <tr key={d.id} className="hover:bg-slate-50">
                <td className="px-4 py-3 text-sm font-medium text-slate-900">{d.name}</td>
                <td className="px-4 py-3">
                  <span className={`text-xs px-2 py-0.5 rounded-full ${
                    d.origin === 'uploaded' ? 'bg-blue-100 text-blue-700' : 'bg-green-100 text-green-700'
                  }`}>{d.origin}</span>
                </td>
                <td className="px-4 py-3 text-sm text-slate-600">{d.rows} × {d.columns}</td>
                <td className="px-4 py-3 text-sm text-slate-600 font-mono">{d.label_column || '—'}</td>
                <td className="px-4 py-3 text-sm text-slate-500">{new Date(d.created_at).toLocaleDateString()}</td>
                <td className="px-4 py-3 text-right space-x-1">
                  <button onClick={() => handleView(d.id)} className="p-1.5 text-slate-400 hover:text-cyan-600 rounded"><Eye className="w-4 h-4" /></button>
                  <a href={`/api/datasets/${d.id}/download`} className="p-1.5 text-slate-400 hover:text-cyan-600 rounded inline-block"><Download className="w-4 h-4" /></a>
                  <button onClick={() => handleDelete(d.id)} className="p-1.5 text-slate-400 hover:text-red-500 rounded"><Trash2 className="w-4 h-4" /></button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Preview panel */}
      {selected && (
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-slate-900">Preview: {selected.meta.name}</h3>
            <button onClick={() => setSelected(null)} className="text-sm text-slate-400 hover:text-slate-600">Close</button>
          </div>
          <div className="flex gap-4 mb-4 text-sm">
            <span className="text-slate-500">Rows: <strong className="text-slate-900">{selected.meta.rows}</strong></span>
            <span className="text-slate-500">Columns: <strong className="text-slate-900">{selected.meta.columns}</strong></span>
            <span className="text-slate-500">Label: <strong className="text-slate-900 font-mono">{selected.meta.label_column || '—'}</strong></span>
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200">
                  {selected.meta.column_names.map(col => (
                    <th key={col} className="px-3 py-2 text-left font-medium text-slate-600 whitespace-nowrap">{col}
                      <span className="block text-xs text-slate-400 font-normal">{selected.dtypes[col]}</span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {selected.head.slice(0, 10).map((row, i) => (
                  <tr key={i} className="border-b border-slate-100">
                    {selected!.meta.column_names.map(col => (
                      <td key={col} className="px-3 py-1.5 text-slate-700 whitespace-nowrap">{String(row[col] ?? '')}</td>
                    ))}
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
