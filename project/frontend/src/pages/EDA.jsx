import { BarChart2, TrendingUp, AlertTriangle, Database } from 'lucide-react';
import { edaMetrics } from '../data/mockData';

const EDA = () => {
  return (
    <div className="scroll-container flex-col gap-6">
      <header className="flex-row justify-between items-end animate-in delay-1">
        <div className="flex-col gap-1">
          <span className="label text-cyan">Exploratory Data Analysis</span>
          <h1 className="gradient-text">Feature Drift & Insights</h1>
        </div>
      </header>

      <div className="grid-3 animate-in delay-2">
        <div className="glass-card p-6 flex-col gap-4" style={{ borderTop: '2px solid var(--violet)' }}>
          <div className="flex-row items-center gap-3">
            <TrendingUp className="text-violet" />
            <h3 className="m-0">Temporal Shift Detected</h3>
          </div>
          <p className="text-secondary text-sm m-0">Recent transaction batches show a 15% increase in night-time volume. Consider retraining models to weight this feature lower.</p>
        </div>
        <div className="glass-card p-6 flex-col gap-4" style={{ borderTop: '2px solid var(--amber)' }}>
          <div className="flex-row items-center gap-3">
            <AlertTriangle className="text-amber" />
            <h3 className="m-0">IP Distance Variance</h3>
          </div>
          <p className="text-secondary text-sm m-0">Subnet matching logic is throwing more nulls in the latest batch. Check ingestion pipeline upstream for IP formatting issues.</p>
        </div>
        <div className="glass-card p-6 flex-col gap-4" style={{ borderTop: '2px solid var(--emerald)' }}>
          <div className="flex-row items-center gap-3">
            <Database className="text-emerald" />
            <h3 className="m-0">Dataset Integrity</h3>
          </div>
          <p className="text-secondary text-sm m-0">q3_transactions_v2.csv passed all schema validation rules. No missing values in critical columns.</p>
        </div>
      </div>

      <div className="glass-card p-6 flex-col gap-6 animate-in delay-3">
        <h3 className="flex items-center gap-2"><BarChart2 size={16} className="text-cyan"/> Feature Importance & Drift Matrix</h3>
        <table>
          <thead>
            <tr>
              <th>Feature Name</th>
              <th>XGBoost Importance (Weight)</th>
              <th>Distribution Drift (7d)</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {edaMetrics.map((metric, i) => (
              <tr key={i}>
                <td className="font-medium">{metric.feature}</td>
                <td>
                  <div className="flex-row items-center gap-3">
                    <span style={{ width: '40px' }}>{metric.importance.toFixed(2)}</span>
                    <div className="progress-bar" style={{ flex: 1, maxWidth: '100px' }}>
                      <div className="progress-fill" style={{ width: `${metric.importance * 100}%`, background: 'var(--cyan)' }} />
                    </div>
                  </div>
                </td>
                <td className="mono text-sm" style={{ color: metric.drift.startsWith('+') ? 'var(--rose)' : metric.drift.startsWith('-') ? 'var(--emerald)' : 'var(--text-muted)' }}>
                  {metric.drift}
                </td>
                <td>
                  {parseFloat(metric.drift) > 4 ? <span className="badge badge-rose">Drifting</span> : <span className="badge badge-emerald">Stable</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default EDA;
