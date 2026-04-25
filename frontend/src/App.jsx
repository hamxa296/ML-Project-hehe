import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Landing from './pages/Landing';
import PipelineLog from './pages/PipelineLog';
import EDA from './pages/EDA';
import Simulator from './pages/Simulator';
import RunDetail from './pages/RunDetail';
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
            <div className="flex-row w-full h-full">
              <Sidebar />
              <main className="flex-1" style={{ position: 'relative' }}>
                <Routes>
                  <Route path="/dashboard" element={<Dashboard />} />
                  <Route path="/pipeline" element={<PipelineLog />} />
                  <Route path="/pipeline/:id" element={<RunDetail />} />
                  <Route path="/eda" element={<EDA />} />
                  <Route path="/simulator" element={<Simulator />} />
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
