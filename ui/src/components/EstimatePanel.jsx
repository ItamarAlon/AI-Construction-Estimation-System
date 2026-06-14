import { useState, useRef } from "react";
import { estimatePdf } from "../api";
import styles from "./EstimatePanel.module.css";

// Change this to control how many PDFs can be queued at once
const PDF_UPLOAD_LIMIT = 5;

export default function EstimatePanel() {
  const [files, setFiles] = useState([]);
  const [results, setResults] = useState([]);
  const [running, setRunning] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  const addFiles = (incoming) => {
    const pdfs = Array.from(incoming).filter((f) =>
      f.name.toLowerCase().endsWith(".pdf")
    );
    setFiles((prev) => {
      const existing = new Set(prev.map((f) => f.name));
      const fresh = pdfs.filter((f) => !existing.has(f.name));
      return [...prev, ...fresh].slice(0, PDF_UPLOAD_LIMIT);
    });
  };

  const removeFile = (name) =>
    setFiles((prev) => prev.filter((f) => f.name !== name));

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    addFiles(e.dataTransfer.files);
  };

  const handleRun = async () => {
    if (!files.length) return;
    setRunning(true);
    setResults([]);
    const out = [];
    for (const file of files) {
      try {
        const data = await estimatePdf(file);
        out.push({ name: file.name, ...data, error: null });
      } catch (err) {
        out.push({ name: file.name, error: err.message });
      }
      setResults([...out]);
    }
    setRunning(false);
  };

  return (
    <section className={styles.panel}>
      <h2 className={styles.title}>Cost Estimation</h2>

      <div
        className={`${styles.dropzone} ${dragOver ? styles.dragOver : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf"
          multiple
          style={{ display: "none" }}
          onChange={(e) => addFiles(e.target.files)}
        />
        <span className={styles.dropIcon}>📄</span>
        <p className={styles.dropText}>
          Drag & drop PDFs here, or <span className={styles.dropLink}>browse</span>
        </p>
        <p className={styles.dropHint}>Up to {PDF_UPLOAD_LIMIT} PDFs</p>
      </div>

      {files.length > 0 && (
        <ul className={styles.fileList}>
          {files.map((f) => (
            <li key={f.name} className={styles.fileItem}>
              <span className={styles.fileName}>{f.name}</span>
              <button
                className={styles.removeFile}
                onClick={() => removeFile(f.name)}
                disabled={running}
              >
                ✕
              </button>
            </li>
          ))}
        </ul>
      )}

      <button
        className={styles.runBtn}
        onClick={handleRun}
        disabled={!files.length || running}
      >
        {running ? "Running…" : "Run Estimation"}
      </button>

      {results.length > 0 && (
        <div className={styles.results}>
          <h3 className={styles.resultsTitle}>Results</h3>
          {results.map((r) => (
            <div key={r.name} className={styles.resultCard}>
              <p className={styles.resultFile}>{r.name}</p>
              {r.error ? (
                <p className={styles.resultError}>{r.error}</p>
              ) : (
                <>
                  <details className={styles.details}>
                    <summary className={styles.summary}>Detected tasks</summary>
                    <pre className={styles.pre}>{r.detected_tasks}</pre>
                  </details>
                  <pre className={styles.resultText}>{r.result}</pre>
                </>
              )}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
