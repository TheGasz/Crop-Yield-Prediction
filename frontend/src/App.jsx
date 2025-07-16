import React, { useState } from "react";
import "./App.css";

function App() {
  const [formData, setFormData] = useState({
    Soil_Quality: "",
    Seed_Variety: "",
    Fertilizer_Ammount_kg_per_hectare: "",
    Sunny_Days: "",
    Rainfall_mm: "",
    Irrigation_Schedule: "",
  });

  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevState) => ({
      ...prevState,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setPrediction(null);
    setError("");

    try {
      const response = await fetch("http://127.0.0.1:5000/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(
          Object.fromEntries(
            Object.entries(formData).map(([key, value]) => [
              key,
              parseFloat(value),
            ])
          )
        ),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || "Terjadi kesalahan pada server");
      }

      setPrediction(result.predicted_yield_kg_per_hectare);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Prediksi Hasil Panen</h1>
        <p>
          Masukkan data pertanian untuk memprediksi hasil panen (kg per hektar)
        </p>

        <form onSubmit={handleSubmit} className="prediction-form">
          <div className="form-grid">
            {Object.keys(formData).map((key) => (
              <div className="input-group" key={key}>
                <label htmlFor={key}>{key.replace(/_/g, " ")}</label>
                <input
                  type="number"
                  id={key}
                  name={key}
                  value={formData[key]}
                  onChange={handleChange}
                  required
                  step="any"
                />
              </div>
            ))}
          </div>
          <button type="submit" disabled={isLoading}>
            {isLoading ? "Memprediksi..." : "Prediksi Hasil Panen"}
          </button>
        </form>

        {prediction !== null && (
          <div className="result">
            <h2>Hasil Prediksi:</h2>
            <p>{prediction.toFixed(2)} kg/hektar</p>
          </div>
        )}

        {error && (
          <div className="error">
            <h2>Error:</h2>
            <p>{error}</p>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
