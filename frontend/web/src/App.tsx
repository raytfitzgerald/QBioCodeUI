import { Routes, Route } from 'react-router-dom';
import AppShell from './components/layout/AppShell';
import Dashboard from './pages/Dashboard';
import DatasetManager from './pages/DatasetManager';
import DataGeneration from './pages/DataGeneration';
import ComplexityEvaluation from './pages/ComplexityEvaluation';
import Embeddings from './pages/Embeddings';
import ModelTraining from './pages/ModelTraining';
import QProfiler from './pages/QProfiler';
import QSage from './pages/QSage';
import JobMonitor from './pages/JobMonitor';
import QuantumSettings from './pages/QuantumSettings';

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Dashboard />} />
        <Route path="datasets" element={<DatasetManager />} />
        <Route path="generate" element={<DataGeneration />} />
        <Route path="evaluate" element={<ComplexityEvaluation />} />
        <Route path="embeddings" element={<Embeddings />} />
        <Route path="train" element={<ModelTraining />} />
        <Route path="profiler" element={<QProfiler />} />
        <Route path="sage" element={<QSage />} />
        <Route path="jobs" element={<JobMonitor />} />
        <Route path="settings" element={<QuantumSettings />} />
      </Route>
    </Routes>
  );
}
