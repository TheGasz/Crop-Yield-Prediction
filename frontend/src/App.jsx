import React, { useState, useRef, useEffect } from "react";
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
  const [chatMessages, setChatMessages] = useState([
    {
      type: "bot",
      message:
        "🌾 Halo! Saya asisten AI untuk prediksi hasil panen. Silakan isi data pertanian Anda dan saya akan memberikan prediksi serta rekomendasi untuk meningkatkan hasil panen!",
    },
  ]);
  const [chatInput, setChatInput] = useState("");
  const [recommendations, setRecommendations] = useState([]);
  const [isChatOpen, setIsChatOpen] = useState(false);
  const chatMessagesRef = useRef(null);

  // Auto scroll chat to bottom
  useEffect(() => {
    if (chatMessagesRef.current) {
      chatMessagesRef.current.scrollTop = chatMessagesRef.current.scrollHeight;
    }
  }, [chatMessages]);

  const generateRecommendations = (data, predictedYield) => {
    const recs = [];

    // Analisis pupuk
    if (data.Fertilizer_Ammount_kg_per_hectare < 30) {
      recs.push({
        title: "🌱 Rekomendasi Pupuk",
        text: `Tingkatkan pupuk menjadi 40-60 kg/hektar untuk hasil optimal. Saat ini: ${data.Fertilizer_Ammount_kg_per_hectare} kg/hektar.`,
      });
    } else if (data.Fertilizer_Ammount_kg_per_hectare > 80) {
      recs.push({
        title: "⚠️ Pupuk Berlebihan",
        text: `Kurangi pupuk ke 60-70 kg/hektar. Pupuk berlebihan dapat merusak tanah. Saat ini: ${data.Fertilizer_Ammount_kg_per_hectare} kg/hektar.`,
      });
    }

    // Analisis irigasi
    if (data.Irrigation_Schedule < 2) {
      recs.push({
        title: "💧 Tingkatkan Irigasi",
        text: `Tambah frekuensi irigasi menjadi 3-4 kali per minggu untuk hasil maksimal. Saat ini: ${data.Irrigation_Schedule} kali.`,
      });
    } else if (data.Irrigation_Schedule > 5) {
      recs.push({
        title: "🚰 Irigasi Berlebihan",
        text: `Kurangi irigasi ke 3-4 kali per minggu. Terlalu banyak air dapat menyebabkan akar busuk. Saat ini: ${data.Irrigation_Schedule} kali.`,
      });
    }

    // Analisis curah hujan
    if (data.Rainfall_mm < 100) {
      recs.push({
        title: "☔ Curah Hujan Rendah",
        text: `Curah hujan rendah (${data.Rainfall_mm}mm). Tingkatkan irigasi atau gunakan mulsa untuk konservasi air.`,
      });
    }

    // Analisis kualitas tanah
    if (data.Soil_Quality < 60) {
      recs.push({
        title: "🌱 Perbaiki Kualitas Tanah",
        text: `Kualitas tanah rendah (${data.Soil_Quality}/100). Tambahkan kompos organik dan lakukan rotasi tanaman.`,
      });
    }

    // Analisis hari cerah
    if (data.Sunny_Days < 20) {
      recs.push({
        title: "☀️ Hari Cerah Kurang",
        text: `Hari cerah terbatas (${data.Sunny_Days} hari). Pertimbangkan varietas tanaman yang tahan naungan.`,
      });
    }

    // Rekomendasi umum berdasarkan prediksi
    if (predictedYield < 500) {
      recs.push({
        title: "📈 Tingkatkan Hasil Panen",
        text: "Hasil prediksi rendah. Fokus pada perbaikan pupuk organik, irigasi teratur, dan pilih varietas unggul.",
      });
    } else if (predictedYield > 800) {
      recs.push({
        title: "🎉 Hasil Excellent!",
        text: "Prediksi hasil sangat baik! Pertahankan praktik pertanian saat ini dan monitor kondisi tanaman.",
      });
    }

    return recs;
  };

  const addChatMessage = (type, message) => {
    setChatMessages((prev) => [...prev, { type, message }]);
  };

  const toggleChat = () => {
    setIsChatOpen(!isChatOpen);
  };

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
    setRecommendations([]);

    // Add user message to chat
    addChatMessage(
      "user",
      "🔍 Meminta prediksi hasil panen dengan data yang telah diisi..."
    );

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

      const predictedYield = result.predicted_yield_kg_per_hectare;
      setPrediction(predictedYield);

      // Generate recommendations
      const recs = generateRecommendations(
        Object.fromEntries(
          Object.entries(formData).map(([key, value]) => [
            key,
            parseFloat(value),
          ])
        ),
        predictedYield
      );
      setRecommendations(recs);

      // Add bot response to chat
      addChatMessage(
        "bot",
        `✅ Prediksi selesai! Hasil panen diprediksi: ${predictedYield.toFixed(
          2
        )} kg/hektar.\n\n` +
          `📊 Analisis:\n` +
          `• Kualitas tanah: ${formData.Soil_Quality}/100\n` +
          `• Pupuk: ${formData.Fertilizer_Ammount_kg_per_hectare} kg/hektar\n` +
          `• Irigasi: ${formData.Irrigation_Schedule} kali/minggu\n` +
          `• Curah hujan: ${formData.Rainfall_mm} mm\n\n` +
          `🎯 Lihat rekomendasi di bawah untuk meningkatkan hasil panen!`
      );
    } catch (err) {
      setError(err.message);
      addChatMessage("bot", `❌ Maaf, terjadi kesalahan: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleChatSubmit = (e) => {
    e.preventDefault();
    if (!chatInput.trim()) return;

    // Add user message
    addChatMessage("user", chatInput);

    // Simple chatbot responses
    const input = chatInput.toLowerCase();
    let botResponse = "";

    if (input.includes("pupuk") || input.includes("fertilizer")) {
      botResponse =
        "🌱 Untuk pupuk optimal:\n• Tanah subur: 40-60 kg/hektar\n• Tanah kurang subur: 60-80 kg/hektar\n• Gunakan pupuk organik + NPK\n• Aplikasi bertahap sesuai fase pertumbuhan";
    } else if (input.includes("irigasi") || input.includes("air")) {
      botResponse =
        "💧 Tips irigasi terbaik:\n• Frekuensi: 3-4 kali/minggu\n• Waktu: pagi/sore hari\n• Hindari overwatering\n• Monitor kelembaban tanah\n• Gunakan sistem tetes jika memungkinkan";
    } else if (input.includes("tanah") || input.includes("soil")) {
      botResponse =
        "🌱 Untuk meningkatkan kualitas tanah:\n• Tambahkan kompos organik\n• Lakukan rotasi tanaman\n• Cek pH tanah (6.0-7.0 optimal)\n• Hindari pengolahan berlebihan\n• Gunakan cover crop";
    } else if (input.includes("varietas") || input.includes("benih")) {
      botResponse =
        "🌾 Pemilihan varietas benih:\n• Sesuaikan dengan iklim lokal\n• Pilih varietas tahan hama\n• Pertimbangkan masa panen\n• Gunakan benih bersertifikat\n• Varietas unggul lokal biasanya lebih adaptif";
    } else if (input.includes("cuaca") || input.includes("iklim")) {
      botResponse =
        "🌤️ Adaptasi dengan cuaca:\n• Pantau prakiraan cuaca\n• Siapkan drainase untuk musim hujan\n• Gunakan mulsa saat kemarau\n• Pertimbangkan greenhouse kecil\n• Sesuaikan jadwal tanam dengan musim";
    } else {
      botResponse =
        "🤖 Saya bisa membantu dengan:\n• Tips pupuk dan pemupukan\n• Strategi irigasi\n• Perbaikan kualitas tanah\n• Pemilihan varietas benih\n• Adaptasi cuaca\n\nTanyakan apa saja seputar pertanian!";
    }

    setTimeout(() => {
      addChatMessage("bot", botResponse);
    }, 1000);

    setChatInput("");
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1 className="main-title">🌾 AgriPredict AI</h1>
        <p className="subtitle">
          Sistem Prediksi Hasil Panen Berbasis AI dengan Rekomendasi Cerdas
        </p>

        <div className="dashboard">
          {/* Prediction Section */}
          <div className="prediction-section">
            <h2 className="section-title">📊 Input Data Pertanian</h2>
            <form onSubmit={handleSubmit} className="prediction-form">
              <div className="form-grid">
                {Object.keys(formData).map((key) => (
                  <div className="input-group" key={key}>
                    <label htmlFor={key}>
                      {key.replace(/_/g, " ").replace("Ammount", "Amount")}
                    </label>

                    {key === "Seed_Variety" ? (
                      // Dropdown untuk Seed Variety (0 atau 1)
                      <select
                        id={key}
                        name={key}
                        value={formData[key]}
                        onChange={handleChange}
                        required
                        className="select-input"
                      >
                        <option value="">Pilih Varietas Benih</option>
                        <option value="0">Varietas 0 (Standar)</option>
                        <option value="1">Varietas 1 (Unggul)</option>
                      </select>
                    ) : key === "Soil_Quality" ? (
                      // Input number untuk Soil Quality dengan batasan 50-100
                      <input
                        type="number"
                        id={key}
                        name={key}
                        min="50"
                        max="100"
                        value={formData[key]}
                        onChange={handleChange}
                        required
                        placeholder="Masukkan kualitas tanah (50-100)"
                      />
                    ) : (
                      // Input biasa untuk field lainnya
                      <input
                        type="number"
                        id={key}
                        name={key}
                        value={formData[key]}
                        onChange={handleChange}
                        required
                        step="any"
                        placeholder={`Masukkan ${key
                          .replace(/_/g, " ")
                          .toLowerCase()}`}
                      />
                    )}
                  </div>
                ))}
              </div>
              <button
                type="submit"
                disabled={isLoading}
                className="predict-button"
              >
                {isLoading ? "🔄 Memproses..." : "🚀 Prediksi Hasil Panen"}
              </button>
            </form>

            {prediction !== null && (
              <div className="result">
                <h2>🎯 Hasil Prediksi:</h2>
                <div className="result-value">
                  {prediction.toFixed(2)} kg/hektar
                </div>
              </div>
            )}

            {error && (
              <div className="error">
                <h2>❌ Error:</h2>
                <p>{error}</p>
              </div>
            )}

            {recommendations.length > 0 && (
              <div className="recommendations">
                <h3>💡 Rekomendasi Peningkatan Hasil Panen</h3>
                {recommendations.map((rec, index) => (
                  <div key={index} className="recommendation-item">
                    <div className="recommendation-title">{rec.title}</div>
                    <div className="recommendation-text">{rec.text}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Floating Chat Button */}
        <button
          className={`chat-toggle-button ${isChatOpen ? "active" : ""}`}
          onClick={toggleChat}
          title="Chat dengan AI Pertanian"
        >
          {isChatOpen ? "✕" : "🤖"}
        </button>

        {/* Chat Popup */}
        <div className={`chat-popup ${isChatOpen ? "show" : ""}`}>
          <div className="chat-popup-header">
            <span>🤖 Asisten AI Pertanian</span>
            <button className="chat-popup-close" onClick={toggleChat}>
              ✕
            </button>
          </div>
          <div className="chatbot-section">
            <div className="chat-container">
              <div className="chat-messages" ref={chatMessagesRef}>
                {chatMessages.map((msg, index) => (
                  <div key={index} className={`message ${msg.type}`}>
                    {msg.message.split("\n").map((line, i) => (
                      <div key={i}>{line}</div>
                    ))}
                  </div>
                ))}
              </div>
              <form
                onSubmit={handleChatSubmit}
                className="chat-input-container"
              >
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder="Tanya tentang pupuk, irigasi, tanah..."
                  className="chat-input"
                />
                <button type="submit" className="send-button">
                  📤
                </button>
              </form>
            </div>
          </div>
        </div>
      </header>
    </div>
  );
}

export default App;
