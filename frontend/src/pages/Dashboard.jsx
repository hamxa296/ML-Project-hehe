import { Activity, Database, GitCommit, Play, CheckCircle2, XCircle, Clock } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { pipelineRuns, performanceData, driftData } from '../data/mockData';

const Dashboard = () => {
  return (
    <div className="scroll-container flex-col gap-8">
      <header className="flex-col gap-1 animate-in delay-1">
        <span className="label text-violet">System Overview</span>
        <h1 className="gradient-text-violet">ML Pipeline Dashboard</h1>
        <p className="text-secondary mt-2" style={{ maxWidth: '600px' }}>
          Real-time view of model training, evaluation, and deployment lifecycle.
        </p>
      </header>

      {/* KPI Section */}
      <div className="grid-4 animate-in delay-2">
        {[
          { label: 'Active Pipelines', val: '3', icon: <Play size={20}/>, color: 'var(--violet)' },
          { label: 'Avg Latency', val: '42ms', icon: <Activity size={20}/>, color: 'var(--cyan)' },
          { label: 'Data Processed', val: '1.2TB', icon: <Database size={20}/>, color: 'var(--emerald)' },
          { label: 'Total Commits', val: '842', icon: <GitCommit size={20}/>, color: 'var(--amber)' }
        ].map((kpi, i) => (
          <div key={i} className="glass-card p-6 flex-col gap-4">
            <div className="flex-row items-center gap-3">
              <div style={{ color: kpi.color, padding: '0.5rem', background: `color-mix(in srgb, ${kpi.color} 10%, transparent)`, borderRadius: '8px' }}>
                {kpi.icon}
              </div>
              <span className="label" style={{ color: 'var(--text-secondary)' }}>{kpi.label}</span>
            </div>
            <div className="stat-number">{kpi.val}</div>
          </div>
        ))}
      </div>

      <div className="grid-2 animate-in delay-3">
        {/* Chart 1 */}
        <div className="glass-card p-6 flex-col gap-6" style={{ height: '350px' }}>
          <h3 className="flex items-center gap-2"><Activity size={16} className="text-cyan"/> Model Accuracy Over Epochs</h3>
          <div className="flex-1">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={performanceData}>
                <defs>
                  <linearGradient id="colorAcc" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--cyan)" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="var(--cyan)" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
                <XAxis dataKey="epoch" stroke="var(--text-muted)" tick={{fontSize: 12}} axisLine={false} tickLine={false} />
                <YAxis stroke="var(--text-muted)" tick={{fontSize: 12}} axisLine={false} tickLine={false} />
                <Tooltip 
                  contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-default)', borderRadius: '8px' }}
                  itemStyle={{ color: 'var(--text-primary)' }}
                />
                <Area type="monotone" dataKey="accuracy" stroke="var(--cyan)" strokeWidth={3} fillOpacity={1} fill="url(#colorAcc)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 2 */}
        <div className="glass-card p-6 flex-col gap-6" style={{ height: '350px' }}>
          <h3 className="flex items-center gap-2"><Database size={16} className="text-violet"/> Feature Drift Analysis</h3>
          <div className="flex-1">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={driftData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
                <XAxis dataKey="day" stroke="var(--text-muted)" tick={{fontSize: 12}} axisLine={false} tickLine={false} />
                <YAxis stroke="var(--text-muted)" tick={{fontSize: 12}} axisLine={false} tickLine={false} />
                <Tooltip 
                  cursor={{ fill: 'rgba(255,255,255,0.02)' }}
                  contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-default)', borderRadius: '8px' }}
                />
                <Bar dataKey="score" fill="var(--violet)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* CI/CD Pipeline Runs Table */}
      <div className="glass-card p-6 flex-col gap-6 animate-in delay-4">
        <h3 className="flex items-center gap-2"><GitCommit size={16} className="text-emerald"/> Recent Pipeline Runs</h3>
        <div style={{ overflowX: 'auto' }}>
          <table>
            <thead>
              <tr>
                <th>Run ID</th>
                <th>Model Target</th>
                <th>Status</th>
                <th>Accuracy</th>
                <th>Latency</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              {pipelineRuns.map((run) => (
                <tr key={run.id}>
                  <td className="mono text-violet font-medium">{run.id}</td>
                  <td className="font-medium">{run.model}</td>
                  <td>
                    {run.status === 'success' && <span className="badge badge-emerald"><CheckCircle2 size={12}/> Success</span>}
                    {run.status === 'failed' && <span className="badge badge-rose"><XCircle size={12}/> Failed</span>}
                    {run.status === 'running' && <span className="badge badge-amber"><Clock size={12}/> Running</span>}
                  </td>
                  <td>{run.accuracy ? `${run.accuracy}%` : '-'}</td>
                  <td>{run.latency || '-'}</td>
                  <td className="text-secondary">{run.time}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
