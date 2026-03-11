import { useEffect, useState, useCallback, useRef } from 'react';
import { Upload, Trash2, Download, Eye, Database, Table, ScatterChart as ScatterIcon } from 'lucide-react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { fetchDatasets, uploadDataset, deleteDataset as deleteDs, fetchDataset } from '../api/client';
import type { DatasetMeta, DatasetPreview } from '../types';

const CLASS_COLORS = ['#06b6d4', '#f97316', '#8b5cf6', '#10b981', '#ef4444', '#eab308', '#ec4899', '#14b8a6'];

function getNumericCols(preview: DatasetPreview): string[] {
  const label = preview.meta.label_column;
  return preview.meta.column_names.filter(
    c => c !== label && ['float64', 'float32', 'int64', 'int32', 'int16', 'int8'].includes(preview.dtypes[c])
  );
}

function DatasetPlot({ preview }: { preview: DatasetPreview }) {
  const numCols = getNumericCols(preview);
  if (numCols.length < 2) {
    return (
      <div className="flex items-center justify-center h-48 text-slate-400 text-sm">
        Need at least 2 numeric feature columns to plot a scatter chart.
      </div>
    );
  }

  const xCol = numCols[0];
  const yCol = numCols[1];
  const labelCol = preview.meta.label_column;

  // Group points by class label
  const groups: Record<string, { x: number; y: number }[]> = {};
  for (const row of preview.head) {
    const cls = labelCol ? String(row[labelCol] ?? 'unknown') : 'all';
    if (!groups[cls]) groups[cls] = [];
    groups[cls].push({ x: Number(row[xCol]), y: Number(row[yCol]) });
  }

  const classNames = Object.keys(groups).sort();

  return (
    <ResponsiveContainer width="100%" height={360}>
      <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis type="number" dataKey="x" name={xCol} tick={{ fontSize: 12 }} label={{ value: xCol, position: 'insideBottom', offset: -10, style: { fontSize: 13, fill: '#64748b' } }} />
        <YAxis type="number" dataKey="y" name={yCol} tick={{ fontSize: 12 }} label={{ value: yCol, angle: -90, position: 'insideLeft', offset: 10, style: { fontSize: 13, fill: '#64748b' } }} />
        <Tooltip cursor={{ strokeDasharray: '3 3' }} formatter={(value: number) => value.toFixed(4)} />
        <Legend verticalAlign="top" />
        {classNames.map((cls, i) => (
          <Scatter
            key={cls}
            name={`Class ${cls}`}
            data={groups[cls]}
            fill={CLASS_COLORS[i % CLASS_COLORS.length]}
            opacity={0.75}
          />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  );
}

export default function DatasetManager() {
  const [datasets, setDatasets] = useState<DatasetMeta[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [selected, setSelected] = useState<DatasetPreview | null>(null);
  const previewRef = useRef<HTMLDivElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [previewTab, setPreviewTab] = useState<'plot' | 'table'>('plot');

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
    // Fetch all rows (rows=0) so the scatter plot shows the full dataset
    const r = await fetchDataset(id, 0);
    setSelected(r.data);
    setPreviewTab('plot');
    setTimeout(() => previewRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50);
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
        <div ref={previewRef} className="bg-white rounded-xl border border-slate-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-slate-900">Preview: {selected.meta.name}</h3>
            <button onClick={() => setSelected(null)} className="text-sm text-slate-400 hover:text-slate-600">Close</button>
          </div>
          <div className="flex gap-4 mb-4 text-sm">
            <span className="text-slate-500">Rows: <strong className="text-slate-900">{selected.meta.rows}</strong></span>
            <span className="text-slate-500">Columns: <strong className="text-slate-900">{selected.meta.columns}</strong></span>
            <span className="text-slate-500">Label: <strong className="text-slate-900 font-mono">{selected.meta.label_column || '—'}</strong></span>
          </div>

          {/* Tab toggle */}
          <div className="flex gap-1 mb-4 bg-slate-100 rounded-lg p-1 w-fit">
            <button
              onClick={() => setPreviewTab('plot')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                previewTab === 'plot' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-700'
              }`}
            >
              <ScatterIcon className="w-4 h-4" /> Plot
            </button>
            <button
              onClick={() => setPreviewTab('table')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                previewTab === 'table' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-700'
              }`}
            >
              <Table className="w-4 h-4" /> Table
            </button>
          </div>

          {previewTab === 'plot' ? (
            <DatasetPlot preview={selected} />
          ) : (
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
                  {selected.head.slice(0, 20).map((row, i) => (
                    <tr key={i} className="border-b border-slate-100">
                      {selected!.meta.column_names.map(col => (
                        <td key={col} className="px-3 py-1.5 text-slate-700 whitespace-nowrap">{String(row[col] ?? '')}</td>
                      ))}
                    </tr>
                  ))}
                  {selected.head.length > 20 && (
                    <tr>
                      <td colSpan={selected.meta.column_names.length} className="px-3 py-2 text-center text-xs text-slate-400">
                        Showing 20 of {selected.head.length} rows
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
