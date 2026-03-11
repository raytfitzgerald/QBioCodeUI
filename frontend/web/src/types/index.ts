export interface DatasetMeta {
  id: string;
  name: string;
  filename: string;
  origin: 'uploaded' | 'generated';
  rows: number;
  columns: number;
  column_names: string[];
  created_at: string;
  label_column: string | null;
}

export interface DatasetPreview {
  meta: DatasetMeta;
  head: Record<string, unknown>[];
  dtypes: Record<string, string>;
  stats?: Record<string, unknown>;
}

export interface GeneratorType {
  type: string;
  label: string;
  description: string;
  params: string[];
}

export interface Job {
  id: string;
  type: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  message: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  result?: Record<string, unknown>;
  error?: string;
}

export interface ModelSchema {
  label: string;
  category: 'classical' | 'quantum';
  params: Record<string, ParamSchema>;
}

export interface ParamSchema {
  type: string;
  default: unknown;
  description?: string;
  options?: string[];
}

export interface EmbeddingMethod {
  method: string;
  label: string;
  description: string;
  params: string[];
}

export interface EvaluationMetric {
  key: string;
  label: string;
  description: string;
}

export interface ProfilerRun {
  run_id: string;
  dataset_ids: string[];
  models: string[];
  embeddings: string[];
  created_at: string;
  status: string;
  n_model_results: number;
  n_eval_results: number;
}

export interface SageModel {
  sage_id: string;
  profiler_run_id: string;
  sage_type: string;
  created_at: string;
}

export interface TrainResult {
  model: string;
  accuracy: number;
  f1_score: number;
  auc?: number;
  time: number;
  params: Record<string, unknown>;
}
