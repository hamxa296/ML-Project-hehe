import { NavLink } from 'react-router-dom';
import { LayoutDashboard, GitMerge, BarChart2, PlayCircle, Settings, Activity } from 'lucide-react';

const Sidebar = () => {
  return (
    <div className="glass-card flex-col animate-in" style={{ width: '260px', height: 'calc(100vh - 40px)', margin: '20px', padding: '1.5rem', zIndex: 10 }}>
      <div className="flex-col gap-2 mb-8">
        <h2 className="flex items-center gap-2 text-primary" style={{ margin: 0 }}>
          <Activity size={24} className="text-violet" /> 
          <span>MLxFlow</span>
        </h2>
        <div className="flex items-center gap-2">
          <div className="live-dot" />
          <span className="label">Cluster Online</span>
        </div>
      </div>

      <nav className="flex-col gap-2 flex-1">
        <NavLink to="/dashboard" className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}>
          <LayoutDashboard size={18} /> Overview
        </NavLink>
        <NavLink to="/pipeline" className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}>
          <GitMerge size={18} /> Pipeline Runs
        </NavLink>
        <NavLink to="/eda" className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}>
          <BarChart2 size={18} /> EDA Insights
        </NavLink>
        <NavLink to="/simulator" className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}>
          <PlayCircle size={18} /> Model Simulator
        </NavLink>
      </nav>

      <div style={{ paddingTop: '1.5rem', borderTop: '1px solid var(--border-subtle)' }}>
        <div className="sidebar-link" style={{ cursor: 'not-allowed', opacity: 0.5 }}>
          <Settings size={18} /> Settings
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
