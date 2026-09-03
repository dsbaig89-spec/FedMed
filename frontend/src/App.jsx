import { useEffect, useState } from "react";
import axios from "axios";
import "./App.css";

const API = "http://127.0.0.1:8000";

function App() {
  const [metrics, setMetrics] = useState(null);
  const [hospitals, setHospitals] = useState(null);
  const [privacy, setPrivacy] = useState(null);
  const [training, setTraining] = useState(null);

  useEffect(() => {
    axios.get(`${API}/metrics`).then((res) => setMetrics(res.data));
    axios.get(`${API}/hospitals`).then((res) => setHospitals(res.data));
    axios.get(`${API}/privacy/status`).then((res) => setPrivacy(res.data));
    axios.get(`${API}/training/status`).then((res) => setTraining(res.data));
  }, []);

  return (
    <div className="dashboard">
      <header>
        <h1>FedMed</h1>
        <p>Privacy-Preserving Federated Medical Image Segmentation</p>
      </header>

      <section className="cards">
        <div className="card">
          <h3>Training Status</h3>
          <strong>{training?.status || "Loading..."}</strong>
          <p>{training?.rounds || 0} Federated Rounds</p>
        </div>

        <div className="card">
          <h3>Hospitals</h3>
          <strong>{training?.hospitals || 0}</strong>
          <p>Participating Hospitals</p>
        </div>

        <div className="card">
          <h3>Differential Privacy</h3>
          <strong>{privacy?.differential_privacy ? "Enabled" : "Disabled"}</strong>
          <p>Noise: {privacy?.noise_multiplier ?? "-"}</p>
        </div>
      </section>

      <section className="panel">
        <h2>Hospital Network</h2>

        {hospitals &&
          Object.entries(hospitals).map(([name, data]) => (
            <div className="hospital" key={name}>
              <span>{name}</span>
              <span>{data.status}</span>
              <span>{data.samples} samples</span>
            </div>
          ))}
      </section>

      <section className="panel">
        <h2>Model Performance</h2>

        {metrics && (
          <div className="metrics">
            <div>
              <h3>Centralized</h3>
              <p>Dice: {metrics.centralized.dice}</p>
              <p>IoU: {metrics.centralized.iou}</p>
            </div>

            <div>
              <h3>Federated</h3>
              <p>Dice: {metrics.federated.dice}</p>
              <p>IoU: {metrics.federated.iou}</p>
            </div>

            <div>
              <h3>Federated + DP</h3>
              <p>Dice: {metrics.federated_dp.dice}</p>
              <p>IoU: {metrics.federated_dp.iou}</p>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

export default App;