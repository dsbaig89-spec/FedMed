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
  const [selectedModel, setSelectedModel] = useState(null);

  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(true);
  const [trainingStarted, setTrainingStarted] = useState(false);

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
      setTrainingStarted(true);

      const response = await axios.post(`${API}/training/start`);

      setMessage(response.data.message);

      setTimeout(() => {
        setTrainingStarted(false);
      }, 3000);
    } catch {
      setTrainingStarted(false);
      setMessage("Unable to start training");
    }
  };

  const handleModelClick = (model) => {
    setSelectedModel(
      selectedModel === model ? null : model
    );
  };

  return (
    <div className="dashboard">

      {/* HEADER */}
      <header className="topbar">
        <div>
          <h1>🧠 FedMed</h1>
          <p>Privacy-Preserving Federated Medical AI</p>
        </div>

        <div className="header-actions">

          <span className="api-status">
            ● API Connected
          </span>

          <button
            className="refresh-btn"
            onClick={loadDashboard}
            disabled={loading}
          >
            {loading ? "⟳ Refreshing..." : "↻ Refresh"}
          </button>

          <button
            className="train-btn"
            onClick={startTraining}
            disabled={trainingStarted}
          >
            {trainingStarted
              ? "⏳ Training..."
              : "▶ Start Training"}
          </button>

        </div>
      </header>


      {/* NOTIFICATION */}
      {message && (
        <div className="notification">
          {message}
        </div>
      )}


      {/* HERO */}
      <section className="hero">

        <div>
          <span className="badge">
            FEDERATED LEARNING
          </span>

          <h2>
            Medical AI Training Dashboard
          </h2>

          <p>
            Train MRI segmentation models across hospitals
            without sharing sensitive patient data.
          </p>
        </div>

        <div className="hero-icon">
          🏥
        </div>

      </section>


      {/* SUMMARY CARDS */}
      <section className="cards">

        <div className="card">
          <div className="card-icon">⚡</div>

          <h3>Training Status</h3>

          <strong>
            {loading
              ? "Loading..."
              : training?.status}
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

          <p>
            Participating hospitals
          </p>
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
            Noise multiplier:{" "}
            {privacy?.noise_multiplier ?? "-"}
          </p>
        </div>


        <div className="card">
          <div className="card-icon">💻</div>

          <h3>Compute Device</h3>

          <strong>
            {training?.device || "-"}
          </strong>

          <p>
            Training environment
          </p>
        </div>

      </section>


      {/* HOSPITAL NETWORK */}
      <section className="panel">

        <div className="section-title">
          <div>
            <h2>🏥 Hospital Network</h2>

            <p>
              Select a hospital to view its information
            </p>
          </div>
        </div>


        <div className="hospital-grid">

          {hospitals &&
            Object.entries(hospitals).map(
              ([name, data]) => (

                <button
                  className={`hospital-card ${
                    selectedHospital === name
                      ? "selected"
                      : ""
                  }`}
                  key={name}
                  onClick={() =>
                    setSelectedHospital(name)
                  }
                >

                  <div className="hospital-top">

                    <span className="hospital-icon">
                      🏥
                    </span>

                    <span className="online">
                      ● {data.status}
                    </span>

                  </div>

                  <h3>{name}</h3>

                  <div className="sample-count">
                    {data.samples}

                    <span>
                      {" "}samples
                    </span>
                  </div>

                  <p>
                    Local medical dataset
                  </p>

                </button>
              )
            )}

        </div>


        {/* SELECTED HOSPITAL DETAILS */}
        {selectedHospital &&
          hospitals?.[selectedHospital] && (

            <div className="hospital-detail">

              <div>

                <span className="badge">
                  SELECTED HOSPITAL
                </span>

                <h3>
                  {selectedHospital}
                </h3>

                <p>
                  This hospital participates in
                  federated training. Patient data
                  remains locally stored.
                </p>

              </div>


              <div className="hospital-detail-stats">

                <div>
                  <span>Status</span>

                  <strong>
                    🟢{" "}
                    {hospitals[selectedHospital].status}
                  </strong>
                </div>


                <div>
                  <span>Dataset</span>

                  <strong>
                    {hospitals[selectedHospital].samples}
                    {" "}samples
                  </strong>
                </div>


                <div>
                  <span>Data Sharing</span>

                  <strong>
                    🔒 Protected
                  </strong>
                </div>

              </div>


              <button
                className="refresh-btn"
                onClick={() =>
                  setSelectedHospital(null)
                }
              >
                ✕ Close Details
              </button>

            </div>
          )}

      </section>


      {/* MODEL PERFORMANCE */}
      <section className="panel">

        <div className="section-title">
          <div>

            <h2>
              📊 Model Performance
            </h2>

            <p>
              Click a model to highlight its performance
            </p>

          </div>
        </div>


        {metrics && (

          <div className="performance">

            <button
              className={`metric-card ${
                selectedModel === "centralized"
                  ? "selected"
                  : ""
              }`}
              onClick={() =>
                handleModelClick("centralized")
              }
            >

              <Metric
                title="Centralized"
                dice={metrics.centralized.dice}
                iou={metrics.centralized.iou}
              />

            </button>


            <button
              className={`metric-card ${
                selectedModel === "federated"
                  ? "selected"
                  : ""
              }`}
              onClick={() =>
                handleModelClick("federated")
              }
            >

              <Metric
                title="Federated"
                dice={metrics.federated.dice}
                iou={metrics.federated.iou}
              />

            </button>


            <button
              className={`metric-card ${
                selectedModel === "dp"
                  ? "selected"
                  : ""
              }`}
              onClick={() =>
                handleModelClick("dp")
              }
            >

              <Metric
                title="Federated + DP"
                dice={metrics.federated_dp.dice}
                iou={metrics.federated_dp.iou}
              />

            </button>

          </div>

        )}


        {/* MODEL DETAILS */}
        {selectedModel && metrics && (

          <div className="hospital-detail">

            <div>

              <span className="badge">
                MODEL SELECTED
              </span>

              <h3>
                {selectedModel === "centralized"
                  ? "Centralized Model"
                  : selectedModel === "federated"
                  ? "Federated Model"
                  : "Federated + Differential Privacy"}
              </h3>

              <p>
                Performance details for the selected
                segmentation approach.
              </p>

            </div>


            <div className="hospital-detail-stats">

              <div>
                <span>Dice Score</span>

                <strong>
                  {selectedModel === "centralized"
                    ? metrics.centralized.dice
                    : selectedModel === "federated"
                    ? metrics.federated.dice
                    : metrics.federated_dp.dice}
                </strong>
              </div>


              <div>
                <span>IoU Score</span>

                <strong>
                  {selectedModel === "centralized"
                    ? metrics.centralized.iou
                    : selectedModel === "federated"
                    ? metrics.federated.iou
                    : metrics.federated_dp.iou}
                </strong>
              </div>


              <div>
                <span>Privacy</span>

                <strong>
                  {selectedModel === "dp"
                    ? "🔐 Enabled"
                    : "Standard"}
                </strong>
              </div>

            </div>


            <button
              className="refresh-btn"
              onClick={() =>
                setSelectedModel(null)
              }
            >
              ✕ Close Details
            </button>

          </div>

        )}

      </section>


      {/* PRIVACY */}
      <section className="panel privacy-panel">

        <div>

          <span className="badge">
            PRIVACY
          </span>

          <h2>
            🔐 Privacy Protection
          </h2>

          <p>
            Differential Privacy protects sensitive
            medical information during federated
            model training.
          </p>

        </div>


        <div className="privacy-info">

          <div>
            <span>Max Norm</span>

            <strong>
              {privacy?.max_norm ?? "-"}
            </strong>
          </div>


          <div>
            <span>Noise Multiplier</span>

            <strong>
              {privacy?.noise_multiplier ?? "-"}
            </strong>
          </div>


          <div>
            <span>Protection</span>

            <strong>
              {privacy?.status || "Loading..."}
            </strong>
          </div>

        </div>

      </section>


      {/* FOOTER */}
      <footer>
        FedMed • Federated Medical Image Segmentation
      </footer>

    </div>
  );
}


/* METRIC COMPONENT */
function Metric({ title, dice, iou }) {

  return (
    <>
      <h3>{title}</h3>

      <div className="metric-row">

        <span>
          Dice Score
        </span>

        <strong>
          {dice}
        </strong>

      </div>


      <div className="progress">

        <div
          className="progress-fill"
          style={{
            width: `${dice * 100}%`,
          }}
        />

      </div>


      <div className="metric-row">

        <span>
          IoU Score
        </span>

        <strong>
          {iou}
        </strong>

      </div>


      <div className="progress">

        <div
          className="progress-fill"
          style={{
            width: `${iou * 100}%`,
          }}
        />

      </div>

    </>
  );
}


export default App;