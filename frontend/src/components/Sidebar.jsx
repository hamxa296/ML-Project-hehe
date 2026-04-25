import { NavLink } from 'react-router-dom';
import { LayoutDashboard, List, ShieldCheck, Settings, Activity, SlidersHorizontal, Database } from 'lucide-react';

const Sidebar = () => {
  const activeStyle = {
    background: 'rgba(197, 160, 89, 0.1)',
    color: 'var(--accent)',
    borderLeft: '4px solid var(--accent)',
    fontWeight: '700'
  };
  
  const linkStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: '0.8rem',
    padding: '14px 20px',
    color: 'var(--text-secondary)',
    textDecoration: 'none',
    transition: 'all 0.3s ease',
    fontWeight: '500',
    borderRadius: '8px',
    margin: '4px 0',
    fontSize: '0.9rem'
  };

  return (
    <div className="glass-card animate-slide-up" style={{ width: '280px', height: 'calc(100vh - 40px)', margin: '20px', borderRadius: '16px', display: 'flex', flexDirection: 'column', border: '1px solid var(--border-glass-bright)', zIndex: 10 }}>
      <div style={{ padding: '2.5rem 1.5rem', marginBottom: '1rem' }}>
        <h2 style={{ color: 'white', display: 'flex', alignItems: 'center', gap: '0.8rem', fontSize: '1.3rem', margin: 0 }}>
          <Activity size={24} color="var(--accent)" /> 
          <span style={{ letterSpacing: '-0.02em', fontWeight: '800' }}>Oracle AI</span>
        </h2>
        <div className="flex-row items-center gap-2" style={{ marginTop: '0.6rem' }}>
          <div className="live-dot" style={{ width: '6px', height: '6px', background: 'var(--accent)' }} />
          <span className="text-muted" style={{ fontSize: '0.75rem', fontWeight: '600' }}>Secure Sequence Active</span>
        </div>
      </div>

      <nav style={{ display: 'flex', flexDirection: 'column', flex: 1, padding: '0 1rem' }}>
        <NavLink to="/dashboard" 
          style={({ isActive }) => isActive ? { ...linkStyle, ...activeStyle } : linkStyle}
          className={({ isActive }) => isActive ? "" : "hover:bg-[rgba(255,255,255,0.03)]"}
        >
          <LayoutDashboard size={18} /> System Status
        </NavLink>
        <NavLink to="/simulation" 
          style={({ isActive }) => isActive ? { ...linkStyle, ...activeStyle } : linkStyle}
          className={({ isActive }) => isActive ? "" : "hover:bg-[rgba(255,255,255,0.03)]"}
        >
          <SlidersHorizontal size={18} /> Live Simulator
        </NavLink>
        <NavLink to="/patterns" 
          style={({ isActive }) => isActive ? { ...linkStyle, ...activeStyle } : linkStyle}
          className={({ isActive }) => isActive ? "" : "hover:bg-[rgba(255,255,255,0.03)]"}
        >
          <Database size={18} /> Knowledge Base
        </NavLink>
        <NavLink to="/transactions" 
          style={({ isActive }) => isActive ? { ...linkStyle, ...activeStyle } : linkStyle}
          className={({ isActive }) => isActive ? "" : "hover:bg-[rgba(255,255,255,0.03)]"}
        >
          <List size={18} /> Transactions Log
        </NavLink>
      </nav>

      <div style={{ padding: '1.5rem', borderTop: '1px solid var(--border-glass-bright)' }}>
        <div style={{ ...linkStyle, cursor: 'default', opacity: 0.4 }}>
          <Settings size={18} /> Configuration
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
