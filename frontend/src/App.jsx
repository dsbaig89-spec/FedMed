import { useEffect, useState } from "react";
import axios from "axios";
import "./App.css";

const API = "http://127.0.0.1:8000";

function App() {
  const [metrics, setMetrics] = useState(null);
  const [hospitals, setHospitals] = useState(null);
  const [privacy, setPrivacy] = useState(null);
  const [training, setTraining] = useState(null);
  const [selectedHospital, setSelectedHospital] = useState(null);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(true);

  const loadDashboard = async () => {
    setLoading(true);

    try {
      const [m, h, p, t] = await Promise.all([
        axios.get(`${API}/metrics`),
        axios.get(`${API}/hospitals`),
        axios.get(`${API}/privacy/status`),
        axios.get(`${API}/training/status`),
      ]);

      setMetrics(m.data);
      setHospitals(h.data);
      setPrivacy(p.data);
      setTraining(t.data);
      setMessage("");
    } catch (error) {
      setMessage("Unable to connect to FedMed API");
    }

    setLoading(false);
  };

  useEffect(() => {
    loadDashboard();
  }, []);

  const startTraining = async () => {
    try {
      const response = await axios.post(`${API}/training/start`);
      setMessage(response.data.message);
    } catch {
      setMessage("Unable to start training");
    }
  };

  return (
    <div className="dashboard">

      <header className="topbar">
        <div>
          <h1>🧠 FedMed</h1>
          <p>Privacy-Preserving Federated Medical AI</p>
        </div>

        <div className="header-actions">
          <span className="api-status">
            ● API Connected
          </span>

          <button className="refresh-btn" onClick={loadDashboard}>
            ↻ Refresh
          </button>

          <button className="train-btn" onClick={startTraining}>
            ▶ Start Training
          </button>
        </div>
      </header>

      {message && (
        <div className="notification">
          {message}
        </div>
      )}

      <section className="hero">
        <div>
          <span className="badge">FEDERATED LEARNING</span>
          <h2>Medical AI Training Dashboard</h2>
          <p>
            Train MRI segmentation models across hospitals without
            sharing sensitive patient data.
          </p>
        </div>

        <div className="hero-icon">
          🏥
        </div>
      </section>

      <section className="cards">

        <div className="card">
          <div className="card-icon">⚡</div>
          <h3>Training Status</h3>

          <strong>
            {loading ? "Loading..." : training?.status}
          </strong>

          <p>
            {training?.rounds || 0} federated rounds
          </p>
        </div>

        <div className="card">
          <div className="card-icon">🏥</div>
          <h3>Hospital Network</h3>

          <strong>
            {training?.hospitals || 0}
          </strong>

          <p>Participating hospitals</p>
        </div>

        <div className="card">
          <div className="card-icon">🔐</div>
          <h3>Differential Privacy</h3>

          <strong className="privacy">
            {privacy?.differential_privacy
              ? "Enabled"
              : "Disabled"}
          </strong>

          <p>
            Noise multiplier: {privacy?.noise_multiplier ?? "-"}
          </p>
        </div>

        <div className="card">
          <div className="card-icon">💻</div>
          <h3>Compute Device</h3>

          <strong>
            {training?.device || "-"}
          </strong>

          <p>Training environment</p>
        </div>

      </section>

      <section className="panel">
        <div className="section-title">
          <div>
            <h2>🏥 Hospital Network</h2>
            <p>Select a hospital to view its information</p>
          </div>
        </div>

        <div className="hospital-grid">

          {hospitals &&
            Object.entries(hospitals).map(([name, data]) => (

              <button
                className={`hospital-card ${
                  selectedHospital === name ? "selected" : ""
                }`}
                key={name}
                onClick={() => setSelectedHospital(name)}
              >

                <div className="hospital-top">
                  <span className="hospital-icon">🏥</span>

                  <span className="online">
                    ● {data.status}
                  </span>
                </div>

                <h3>{name}</h3>

                <div className="sample-count">
                  {data.samples}
                  <span> samples</span>
                </div>

                <p>Local medical dataset</p>

              </button>

            ))}

        </div>

        {selectedHospital && hospitals?.[selectedHospital] && (

          <div className="hospital-detail">

            <h3>
              {selectedHospital} Details
            </h3>

            <p>
              <strong>Status:</strong>{" "}
              {hospitals[selectedHospital].status}
            </p>

            <p>
              <strong>Samples:</strong>{" "}
              {hospitals[selectedHospital].samples}
            </p>

            <p>
              Patient data remains at the hospital.
            </p>

          </div>
        )}

      </section>

      <section className="panel">

        <div className="section-title">
          <div>
            <h2>📊 Model Performance</h2>
            <p>Comparison of segmentation approaches</p>
          </div>
        </div>

        {metrics && (

          <div className="performance">

            <Metric
              title="Centralized"
              dice={metrics.centralized.dice}
              iou={metrics.centralized.iou}
            />

            <Metric
              title="Federated"
              dice={metrics.federated.dice}
              iou={metrics.federated.iou}
            />

            <Metric
              title="Federated + DP"
              dice={metrics.federated_dp.dice}
              iou={metrics.federated_dp.iou}
            />

          </div>

        )}

      </section>

      <section className="panel privacy-panel">

        <div>
          <h2>🔐 Privacy Protection</h2>

          <p>
            Differential Privacy protects sensitive medical
            information during federated model training.
          </p>
        </div>

        <div className="privacy-info">

          <div>
            <span>Max Norm</span>
            <strong>{privacy?.max_norm ?? "-"}</strong>
          </div>

          <div>
            <span>Noise Multiplier</span>
            <strong>{privacy?.noise_multiplier ?? "-"}</strong>
          </div>

          <div>
            <span>Protection</span>
            <strong>
              {privacy?.status || "Loading..."}
            </strong>
          </div>

        </div>

      </section>

      <footer>
        FedMed • Federated Medical Image Segmentation
      </footer>

    </div>
  );
}

function Metric({ title, dice, iou }) {
  return (
    <div className="metric-card">

      <h3>{title}</h3>

      <div className="metric-row">
        <span>Dice Score</span>
        <strong>{dice}</strong>
      </div>

      <div className="progress">
        <div
          className="progress-fill"
          style={{ width: `${dice * 100}%` }}
        />
      </div>

      <div className="metric-row">
        <span>IoU Score</span>
        <strong>{iou}</strong>
      </div>

      <div className="progress">
        <div
          className="progress-fill"
          style={{ width: `${iou * 100}%` }}
        />
      </div>

    </div>
  );
}

export default App;