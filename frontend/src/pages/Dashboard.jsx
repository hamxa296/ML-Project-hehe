import { Activity, ShieldCheck, CheckCircle2, Lock, Cpu, Server } from 'lucide-react';
import { modelAgreement } from '../data/mockData';

const Dashboard = () => {
  return (
    <div className="flex-col gap-8 animate-slide-up" style={{ paddingBottom: '3rem' }}>
      <header className="flex-col stagger-1">
        <span className="section-title"><Lock size={16}/> Security Cloud</span>
        <h1 className="text-gradient-animate">System Operations Node</h1>
        <p className="text-secondary" style={{ maxWidth: '600px', marginTop: '0.5rem' }}>
           Viewing high-level system metrics and multi-model consensus status.
        </p>
      </header>

      {/* KPI Section */}
      <div className="grid-cols-3 stagger-2">
        {[
          { label: 'Total Volume', val: '24,912', change: '+12%', icon: <Activity size={20}/>, border: 'var(--accent)', glow: 'var(--accent-glow)' },
          { label: 'Blocked Threats', val: '142', change: '-4%', icon: <ShieldCheck size={20}/>, border: 'var(--severe)', glow: 'var(--severe-glow)' },
          { label: 'Uptime Status', val: '99.9%', change: 'Stable', icon: <CheckCircle2 size={20}/>, border: 'var(--safe)', glow: 'var(--safe-glow)' }
        ].map((kpi, i) => (
          <div key={i} className="glass-card p-8 flex-col gap-3" style={{ borderLeft: `6px solid ${kpi.border}`, transition: 'all 0.4s' }}>
            <div className="flex-row items-center justify-between">
              <div className="flex-row items-center gap-2 text-secondary" style={{ fontSize: '0.9rem', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                {kpi.icon} {kpi.label}
              </div>
              <span style={{ fontSize: '0.8rem', color: kpi.border, background: `rgba(255,255,255,0.02)`, padding: '4px 10px', borderRadius: '10px' }}>{kpi.change}</span>
            </div>
            <div style={{ fontSize: '3rem', fontWeight: '800', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              {kpi.val}
            </div>
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: '2rem' }}>
        
        {/* Feature 10: Multi-Model Consensus Engine */}
        <div className="glass-card p-10 flex-col gap-6 stagger-3">
          <h3 className="section-title flex-row items-center gap-2"><Cpu size={18}/> Multi-Model Consensus (Live)</h3>
          <p className="text-secondary" style={{ fontSize: '0.95rem' }}>
            Real-time agreement overview across distributed neural checks. The system requires majority consensus for automated decisions.
          </p>
          <div className="flex-col gap-4" style={{ marginTop: '1rem' }}>
            {modelAgreement.details.map((d, i) => (
              <div key={i} className="flex-row justify-between items-center" style={{ padding: '1.2rem', background: 'rgba(255,255,255,0.02)', borderRadius: '12px', border: '1px solid var(--border-glass-bright)' }}>
                <span className="text-primary" style={{ fontSize: '1rem', fontWeight: '600' }}>{d.name}</span>
                <div className="flex-row items-center gap-3">
                   <span style={{ fontSize: '0.85rem', color: 'var(--safe)', fontWeight: '700' }}>ONLINE</span>
                   <div className="live-dot" style={{ background: 'var(--safe)' }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* System Health */}
        <div className="glass-card p-10 flex-col gap-6 stagger-4 text-center items-center justify-center">
           <Server size={48} color="var(--safe)" style={{ opacity: 0.8 }} />
           <div className="flex-col gap-2">
              <h3 style={{ margin: 0, fontSize: '1.5rem', fontWeight: '800' }}>All Systems Nominal</h3>
              <p className="text-muted" style={{ margin: 0 }}>Cluster actively processing logic.</p>
           </div>
        </div>

      </div>

    </div>
  );
};

export default Dashboard;
