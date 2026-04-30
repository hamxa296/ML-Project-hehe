import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Landing from './pages/Landing';
import PipelineLog from './pages/PipelineLog';
import EDA from './pages/EDA';
import RunDetail from './pages/RunDetail';
import DatasetEvaluator from './pages/DatasetEvaluator';
import Models from './pages/Models';
import './index.css';

function App() {
  return (
    <Router>
      <div className="flex-row" style={{ width: '100vw', height: '100vh', position: 'relative' }}>
        <div className="app-bg" />
        <div className="app-grid" />
        
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/*" element={
            <div className="flex-row w-full" style={{ height: '100%' }}>
              <Sidebar />
              <main className="flex-1" style={{ position: 'relative', height: '100%', overflow: 'hidden' }}>
                <Routes>
                  <Route path="/dashboard" element={<Dashboard />} />
                  <Route path="/pipeline" element={<PipelineLog />} />
                  <Route path="/pipeline/:id" element={<RunDetail />} />
                  <Route path="/eda" element={<EDA />} />
                  <Route path="/evaluator" element={<DatasetEvaluator />} />
                  <Route path="/models" element={<Models />} />
                </Routes>
              </main>
            </div>
          } />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
