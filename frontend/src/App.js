import { useEffect, useState } from "react";
import "@/App.css";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import axios from "axios";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const Home = () => {
  const helloWorldApi = async () => {
    try {
      const response = await axios.get(`${API}/`);
      console.log(response.data.message);
    } catch (e) {
      console.error(e, `errored out requesting / api`);
    }
  };

  useEffect(() => {
    helloWorldApi();
  }, []);

  return (
    <div>
      <header className="App-header">
        <a
          className="App-link"
          href="https://emergent.sh"
          target="_blank"
          rel="noopener noreferrer"
        >
          <img src="https://avatars.githubusercontent.com/in/1201222?s=120&u=2686cf91179bbafbc7a71bfbc43004cf9ae1acea&v=4" />
        </a>
        <p className="mt-5">Building something incredible ~!</p>
        <Link to="/visualizer">
          <Button className="mt-4">Open CJPT Visualizer</Button>
        </Link>
      </header>
    </div>
  );
};

const Visualizer = () => {
  const [ligoData, setLigoData] = useState(null);
  const [loading, setLoading] = useState(false);

  const generateData = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/ligo/generate`);
      setLigoData(response.data);
    } catch (e) {
      console.error("Failed to generate LIGO data", e);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 p-8">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-4xl font-bold text-white mb-8">🌊 CJPT Visualizer</h1>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <Card className="bg-slate-800/50 border-cyan-500/20">
            <CardHeader>
              <CardTitle className="text-cyan-400">LIGO Data Generator</CardTitle>
            </CardHeader>
            <CardContent>
              <Button 
                onClick={generateData} 
                disabled={loading}
                className="w-full bg-cyan-600 hover:bg-cyan-700"
              >
                {loading ? "Generating..." : "Generate LIGO Data"}
              </Button>
              
              {ligoData && (
                <div className="mt-4 text-white space-y-2">
                  <p><strong>SNR:</strong> {ligoData.snr.toFixed(2)}</p>
                  <p><strong>Detection:</strong> {ligoData.detection ? "✅ YES" : "❌ NO"}</p>
                  <p><strong>Data Points:</strong> {ligoData.frequency.length}</p>
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-cyan-500/20">
            <CardHeader>
              <CardTitle className="text-cyan-400">Quick Stats</CardTitle>
            </CardHeader>
            <CardContent className="text-white">
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span>Phase:</span>
                  <span className="text-cyan-400 font-bold">MINIMAL</span>
                </div>
                <div className="flex justify-between">
                  <span>Δ_KK:</span>
                  <span className="text-cyan-400 font-mono">0.0024</span>
                </div>
                <div className="flex justify-between">
                  <span>G_trap:</span>
                  <span className="text-cyan-400 font-mono">0.85</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <Card className="bg-slate-800/50 border-cyan-500/20">
          <CardHeader>
            <CardTitle className="text-cyan-400">Waveform Display</CardTitle>
          </CardHeader>
          <CardContent>
            {ligoData ? (
              <div className="h-64 bg-slate-900/50 rounded flex items-center justify-center">
                <svg width="100%" height="100%" viewBox="0 0 500 200">
                  <polyline
                    points={ligoData.strain.map((s, i) => 
                      `${i * 500 / ligoData.strain.length},${100 - s * 50}`
                    ).join(' ')}
                    fill="none"
                    stroke="#22d3ee"
                    strokeWidth="2"
                  />
                </svg>
              </div>
            ) : (
              <div className="h-64 bg-slate-900/50 rounded flex items-center justify-center text-slate-500">
                Generate data to see waveform
              </div>
            )}
          </CardContent>
        </Card>

        <div className="mt-6">
          <Link to="/">
            <Button variant="outline" className="border-cyan-500 text-cyan-400 hover:bg-cyan-500/10">
              ← Back to Home
            </Button>
          </Link>
        </div>
      </div>
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/visualizer" element={<Visualizer />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
