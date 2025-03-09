import React, { useState } from "react";
import axios from "axios";
import "./App.css";

const App = () => {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setResults([]);

    try {
      const response = await axios.post("http://127.0.0.1:5000/search", { query });

      setResults(response.data.results);
    } catch (err) {
      setError("Failed to fetch search results. Please try again.");
    }
    setLoading(false);
  };

  return (
    <div className='app-container'>
      <div className='header-container'>
        <div className='header-left-container'>
          <p>CS 121 / IN4MATX 141</p>
        </div>
        <div className='header-right-container'>
          <p>Search Engine</p>
        </div>

      </div>
      <div className='search-container'>

      <div className='output-container'>
          <div className='output-list-container'>
            {loading && <p>Loading...</p>}
            {error && <p style={{ color: "red" }}>{error}</p>}
            {results.length > 0 && results.map((item, index) => (
              <div key={index} className='output-list-item'>
                <a href={item.url} target="_blank" rel="noopener noreferrer">
                  {item.url} (Score: {item.score.toFixed(4)})
                </a>
              </div>
            ))}
          </div>
        </div>
        
        <div className='input-and-button-container'>
          <input
            className='input-container'
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSearch()}  // ✅ Allow Enter to trigger search
            placeholder="Enter search query..."
          />
          <button onClick={handleSearch} className='button-container'>Search</button>
        </div>
      </div>

      <div className='footer-container'>
        <div className='footer-text'>
          <p>© 2025 Final A3 Project</p>
          <p>Created by: Taiki, Ian, and Nathan</p>
          <p>Professor: Iftekhar Ahmed</p>
        </div>
      </div>
    </div>
  );
};

export default App;
