import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard, Database, Sparkles, BarChart3, Cpu,
  Layers, FlaskConical, BrainCircuit, ListChecks, Settings, Atom
} from 'lucide-react';

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/datasets', icon: Database, label: 'Datasets' },
  { to: '/generate', icon: Sparkles, label: 'Generate Data' },
  { to: '/evaluate', icon: BarChart3, label: 'Evaluate Complexity' },
  { to: '/embeddings', icon: Layers, label: 'Embeddings' },
  { to: '/train', icon: Cpu, label: 'Train Models' },
  { to: '/profiler', icon: FlaskConical, label: 'QProfiler' },
  { to: '/sage', icon: BrainCircuit, label: 'QSage' },
  { to: '/jobs', icon: ListChecks, label: 'Jobs' },
  { to: '/settings', icon: Settings, label: 'Quantum Settings' },
];

export default function Sidebar() {
  return (
    <aside className="w-64 bg-slate-900 text-white flex flex-col min-h-screen">
      <div className="p-5 border-b border-slate-700">
        <div className="flex items-center gap-2">
          <Atom className="w-7 h-7 text-cyan-400" />
          <div>
            <h1 className="text-lg font-bold tracking-tight">QBioCode</h1>
            <p className="text-xs text-slate-400">Quantum ML Platform</p>
          </div>
        </div>
      </div>
      <nav className="flex-1 p-3 space-y-1 overflow-y-auto">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-cyan-600/20 text-cyan-400'
                  : 'text-slate-300 hover:bg-slate-800 hover:text-white'
              }`
            }
          >
            <Icon className="w-4.5 h-4.5 shrink-0" />
            {label}
          </NavLink>
        ))}
      </nav>
      <div className="p-4 border-t border-slate-700 text-xs text-slate-500">
        IBM Research &middot; Cleveland Clinic
      </div>
    </aside>
  );
}
