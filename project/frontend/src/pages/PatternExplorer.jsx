import { Database, Activity, TrendingUp, HelpCircle } from 'lucide-react';

const PatternExplorer = () => {
  const insights = [
    { title: 'Temporal Peak', text: 'Fraud intensity predictably peaks between 02:00 AM and 04:00 AM local server time. The model heavily weights this time window negatively during calculations.', icon: <Activity className="text-severe" size={24}/> },
    { title: 'Volume Threshold', text: 'Transactions exceeding $1.2k exhibit a 4.5x higher probability of anomaly when grouped with digital goods categories.', icon: <TrendingUp className="text-accent" size={24}/> },
    { title: 'Velocity Bursts', text: 'Multiple transactions from the same subnet within a 15-minute window strongly correlative to bot-driven enumeration attacks.', icon: <Database className="text-suspicious" size={24}/> },
    { title: 'Anomaly Core', text: 'International merchant origin combined with fast checkout and no session logging is the primary risk driver in the XGBoost tree.', icon: <HelpCircle className="text-safe" size={24}/> }
  ];

  return (
    <div className="flex-col gap-8 animate-slide-up" style={{ paddingBottom: '3rem' }}>
      <header className="flex-col stagger-1">
        <span className="section-title"><Database size={16}/> Knowledge Base</span>
        <h1 className="text-gradient-animate">Fraud Pattern Explorer</h1>
        <p className="text-secondary" style={{ maxWidth: '600px', marginTop: '0.5rem' }}>
          Insights derived from exploratory data analysis and continuous model feature importance metrics.
        </p>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(400px, 1fr))', gap: '2rem' }}>
        {insights.map((insight, i) => (
          <div key={i} className={`glass-card p-8 flex-col gap-4 stagger-${(i + 2)}`} style={{ borderTop: '2px solid var(--border-glass-bright)' }}>
             <div className="flex-row items-start gap-4">
                <div style={{ padding: '1rem', background: 'rgba(255,255,255,0.03)', borderRadius: '12px' }}>
                   {insight.icon}
                </div>
                <div className="flex-col">
                   <h3 style={{ margin: '0 0 0.5rem 0', fontSize: '1.2rem', fontWeight: 600 }}>{insight.title}</h3>
                   <p className="text-secondary" style={{ fontSize: '0.95rem', lineHeight: '1.6', margin: 0 }}>{insight.text}</p>
                </div>
             </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default PatternExplorer;
