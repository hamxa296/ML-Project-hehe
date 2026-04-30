import { useState, useEffect } from 'react';
import { BarChart2, TrendingUp, AlertTriangle, Database, ArrowRight } from 'lucide-react';

const EDA = () => {
  const [runs, setRuns] = useState([]);

  useEffect(() => {
    fetch('http://localhost:8000/model_evaluations')
      .then(res => res.json())
      .then(data => {
         if(data.evaluations) setRuns(data.evaluations.reverse());
      })
      .catch(err => console.error("Failed to load metrics", err));
  }, []);

  return (
    <div className="scroll-container flex-col gap-6">
      <header className="flex-row justify-between items-end animate-in delay-1">
        <div className="flex-col gap-1">
          <span className="label text-cyan">Model Analytics</span>
          <h1 className="gradient-text">Historical Model Metrics</h1>
        </div>
      </header>

      <div className="grid-3 animate-in delay-2">
        <div className="glass-card p-6 flex-col gap-4" style={{ borderTop: '2px solid var(--violet)' }}>
          <div className="flex-row items-center gap-3">
            <TrendingUp className="text-violet" />
            <h3 className="m-0">Performance Tracking</h3>
          </div>
          <p className="text-secondary text-sm m-0">Monitor model degradation over time. A drop in PR-AUC indicates potential concept drift.</p>
        </div>
        <div className="glass-card p-6 flex-col gap-4" style={{ borderTop: '2px solid var(--amber)' }}>
          <div className="flex-row items-center gap-3">
            <AlertTriangle className="text-amber" />
            <h3 className="m-0">Precision vs Recall</h3>
          </div>
          <p className="text-secondary text-sm m-0">Balancing false positives against missed fraud. Historical changes highlight the impact of hyperparameter tuning.</p>
        </div>
        <div className="glass-card p-6 flex-col gap-4" style={{ borderTop: '2px solid var(--emerald)' }}>
          <div className="flex-row items-center gap-3">
            <Database className="text-emerald" />
            <h3 className="m-0">Paper Baseline</h3>
          </div>
          <p className="text-secondary text-sm m-0">The XGBoost paper baseline aims for 0.834 PR-AUC and 0.887 ROC-AUC. Compare versions below.</p>
        </div>
      </div>

      <div className="glass-card p-6 flex-col gap-6 animate-in delay-3">
        <h3 className="flex items-center gap-2"><BarChart2 size={16} className="text-cyan"/> Version Metrics History</h3>
        <table>
          <thead>
            <tr>
              <th>Version</th>
              <th>Time</th>
              <th>PR-AUC</th>
              <th>ROC-AUC</th>
              <th>Precision</th>
              <th>Recall</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run, i) => {
              const pr_auc = (run.auc_pr * 100).toFixed(2);
              const roc_auc = (run.auc_roc * 100).toFixed(2);
              const prec = (run.precision * 100).toFixed(2);
              const rec = (run.recall * 100).toFixed(2);
              
              const isBaselineBeat = run.auc_pr >= 0.834;

              return (
                <tr key={i}>
                  <td className="font-medium mono text-violet">{run.version}</td>
                  <td className="text-secondary text-sm">{new Date(run.timestamp).toLocaleString()}</td>
                  <td>
                    <div className="flex-row items-center gap-2">
                      <span className={isBaselineBeat ? "text-emerald" : "text-amber"}>{pr_auc}%</span>
                      {isBaselineBeat && <ArrowRight size={14} className="text-emerald" style={{ transform: 'rotate(-45deg)' }}/>}
                    </div>
                  </td>
                  <td>{roc_auc}%</td>
                  <td>{prec}%</td>
                  <td>{rec}%</td>
                </tr>
              );
            })}
            {runs.length === 0 && <tr><td colSpan="6" className="text-center p-4 text-muted">No model evaluations found.</td></tr>}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default EDA;
