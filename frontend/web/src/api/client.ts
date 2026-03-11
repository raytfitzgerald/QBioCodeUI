import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 120000,
});

export default api;

// ---- Datasets ----
export const fetchDatasets = () => api.get('/datasets');
export const fetchDataset = (id: string, rows?: number) =>
  api.get(`/datasets/${id}`, { params: rows !== undefined ? { rows } : {} });
export const uploadDataset = (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  return api.post('/datasets/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};
export const deleteDataset = (id: string) => api.delete(`/datasets/${id}`);

// ---- Generation ----
export const fetchGeneratorTypes = () => api.get('/generate/types');
export const generateData = (params: Record<string, unknown>) =>
  api.post('/generate', params);

// ---- Evaluation ----
export const fetchMetricsInfo = () => api.get('/evaluate/metrics');
export const evaluateDataset = (datasetId: string, labelColumn: string) =>
  api.post('/evaluate/complexity', { dataset_id: datasetId, label_column: labelColumn });

// ---- Models ----
export const fetchAvailableModels = () => api.get('/models/available');
export const trainModel = (params: Record<string, unknown>) =>
  api.post('/models/train', params);

// ---- Embeddings ----
export const fetchEmbeddingMethods = () => api.get('/embeddings/methods');
export const computeEmbedding = (params: Record<string, unknown>) =>
  api.post('/embeddings/compute', params);

// ---- Profiler ----
export const fetchProfilerDefaults = () => api.get('/profiler/config/defaults');
export const runProfiler = (params: Record<string, unknown>) =>
  api.post('/profiler/run', params);
export const fetchProfilerRuns = () => api.get('/profiler/runs');
export const fetchProfilerRun = (id: string) => api.get(`/profiler/runs/${id}`);

// ---- Sage ----
export const trainSage = (params: Record<string, unknown>) =>
  api.post('/sage/train', params);
export const predictSage = (params: Record<string, unknown>) =>
  api.post('/sage/predict', params);
export const fetchSageModels = () => api.get('/sage/models');

// ---- Jobs ----
export const fetchJobs = (status?: string) =>
  api.get('/jobs', { params: status ? { status } : {} });
export const fetchJob = (id: string) => api.get(`/jobs/${id}`);
export const cancelJob = (id: string) => api.post(`/jobs/${id}/cancel`);

// ---- Health ----
export const checkHealth = () => api.get('/health');
