import { useState } from 'react';
import { Settings, Shield, CheckCircle2 } from 'lucide-react';

export default function QuantumSettings() {
  const [backend, setBackend] = useState('simulator');
  const [seed, setSeed] = useState(42);
  const [shots, setShots] = useState(1024);
  const [resilLevel, setResilLevel] = useState(1);
  const [ibmToken, setIbmToken] = useState('');
  const [ibmInstance, setIbmInstance] = useState('');
  const [ibmChannel, setIbmChannel] = useState('ibm_quantum');
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    // Store in localStorage for now — in production, this would go to the backend
    localStorage.setItem('qbiocode_quantum_config', JSON.stringify({
      backend, seed, shots, resil_level: resilLevel,
      ibm_token: ibmToken, ibm_instance: ibmInstance, ibm_channel: ibmChannel,
    }));
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
          <Settings className="w-6 h-6 text-slate-600" /> Quantum Settings
        </h1>
        <p className="text-slate-500 mt-1">Configure quantum backend and IBM Quantum credentials</p>
      </div>

      <div className="bg-white rounded-xl border border-slate-200 p-6 space-y-6">
        <div>
          <h3 className="font-semibold text-slate-900 mb-4">Quantum Backend</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Backend</label>
              <select value={backend} onChange={e => setBackend(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-cyan-500">
                <option value="simulator">Simulator (Local)</option>
                <option value="ibm_least">IBM Least Busy</option>
                <option value="ibm_brisbane">IBM Brisbane</option>
                <option value="ibm_osaka">IBM Osaka</option>
                <option value="ibm_kyoto">IBM Kyoto</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Random Seed</label>
              <input type="number" value={seed} onChange={e => setSeed(Number(e.target.value))}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm" />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Shots</label>
              <input type="number" value={shots} onChange={e => setShots(Number(e.target.value))}
                min={100} max={100000} step={100}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm" />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Resilience Level</label>
              <select value={resilLevel} onChange={e => setResilLevel(Number(e.target.value))}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm">
                <option value={0}>0 - No mitigation</option>
                <option value={1}>1 - Basic</option>
                <option value={2}>2 - Advanced</option>
              </select>
            </div>
          </div>
        </div>

        <div className="border-t border-slate-200 pt-6">
          <h3 className="font-semibold text-slate-900 mb-1 flex items-center gap-2">
            <Shield className="w-4 h-4 text-amber-500" /> IBM Quantum Credentials
          </h3>
          <p className="text-sm text-slate-500 mb-4">Required only when using IBM hardware backends</p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Channel</label>
              <select value={ibmChannel} onChange={e => setIbmChannel(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm">
                <option value="ibm_quantum">ibm_quantum</option>
                <option value="ibm_cloud">ibm_cloud</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Instance</label>
              <input type="text" value={ibmInstance} onChange={e => setIbmInstance(e.target.value)}
                placeholder="hub/group/project"
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm" />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">API Token</label>
              <input type="password" value={ibmToken} onChange={e => setIbmToken(e.target.value)}
                placeholder="Your IBM Quantum API token"
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm" />
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <button onClick={handleSave}
            className="px-5 py-2.5 bg-cyan-600 text-white rounded-lg font-medium hover:bg-cyan-700">
            Save Settings
          </button>
          {saved && (
            <span className="flex items-center gap-1 text-sm text-green-600">
              <CheckCircle2 className="w-4 h-4" /> Settings saved
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
