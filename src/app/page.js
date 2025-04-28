"use client";
import { useState } from "react";
import "./App.css"; // Link your CSS file here

export default function App() {
  const [texts, setTexts] = useState({ imdb: '', yelp: '', amazon: '', merged: '', all: '' });
  const [results, setResults] = useState({
    imdb: null,
    yelp: null,
    amazon: null,
    merged: null,
    all_combined: null
  });

  const handlePredict = async (text, endpoint, platform) => {
    try {
      const res = await fetch(`http://127.0.0.1:5000/${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });
      const data = await res.json();
      console.log('API response:', data);
  
      if (platform === "all") {
        // platform === "all" → response has imdb, yelp, amazon, merged inside
        setResults(prev => ({
          ...prev,
          imdb: {
            logreg: data.imdb.logreg,
            bayes: data.imdb.bayes,
            bert: data.imdb.bert
          },
          yelp: {
            logreg: data.yelp.logreg,
            bayes: data.yelp.bayes,
            bert: data.yelp.bert
          },
          amazon: {
            logreg: data.amazon.logreg,
            bayes: data.amazon.bayes,
            bert: data.amazon.bert
          },
          merged: {
            logreg: data.merged.logreg,
            bayes: data.merged.bayes,
            bert: data.merged.bert
          }
        }));
      } else {
        setResults(prev => ({
          ...prev,
          [platform]: {
            logreg: data.logreg,
            bayes: data.bayes,
            bert: data.bert,
            combined: data.combined
          }
        }));
      }
    } catch (error) {
      console.error("Prediction error:", error);
    }
  };
  

  return (
    <div className="container">
      <h1>Sentiment Analysis Web App</h1>

      {['imdb', 'yelp', 'amazon', 'merged'].map(platform => (
        <div key={platform} className="card">
          <h2>{platform.toUpperCase()} - All Models</h2>
          <input
            value={texts[platform]}
            onChange={e => setTexts({ ...texts, [platform]: e.target.value })}
            placeholder={`Enter text for ${platform}`}
          />
          <button onClick={() => handlePredict(texts[platform], `predict-all-${platform}`, platform)}>
            Predict All
          </button>

          {results[platform] && (
            <div className="results">
              <p>Logistic Regression: {results[platform].logreg}</p>
              <p>Naive Bayes: {results[platform].bayes}</p>
              <p>BERT: {results[platform].bert}</p>
              <p><strong>Combined Result:</strong> {results[platform].combined}</p>
            </div>
          )}
        </div>
      ))}

      <div className="card">
        <h2>All Models on All Platforms</h2>
        <input
          value={texts.all}
          onChange={e => setTexts({ ...texts, all: e.target.value })}
          placeholder="Enter text for all platforms"
        />
        <button onClick={() => handlePredict(texts.all, 'predict-all', 'all')}>
          Predict All
        </button>

        {results.all_combined && (
          <div className="results">
            <p><strong>Global Combined Result:</strong> {results.all_combined}</p>
          </div>
        )}
      </div>
    </div>
  );
}
