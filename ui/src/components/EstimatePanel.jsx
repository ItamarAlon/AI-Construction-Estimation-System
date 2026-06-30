import { useState, useRef } from "react";
import { estimatePdf } from "../api";
import styles from "./EstimatePanel.module.css";

function BreakdownTable({ items, totalLabel, total, styles }) {
  const fmt = (n) => n?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  return (
    <table className={styles.breakdownTable}>
      <thead>
        <tr>
          <th className={styles.colTask}>Task</th>
          <th className={styles.colNum}>Qty</th>
          <th className={styles.colNum}>Unit ₪</th>
          <th className={styles.colNum}>Total ₪</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.task} className={styles.breakdownRow}>
            <td className={styles.colTask}>{item.task.replace(/ \(per meter\)$/i, "")}</td>
            <td className={styles.colNum}>
              {item.task.match(/\(per meter\)$/i) ? `${item.quantity}m` : item.quantity}
            </td>
            <td className={styles.colNum}>{item.unit_price?.toLocaleString()}</td>
            <td className={styles.colNum}>{fmt(item.cost)}</td>
          </tr>
        ))}
      </tbody>
      <tfoot>
        <tr className={styles.totalRow}>
          <td colSpan={3} className={styles.totalLabel}>{totalLabel}</td>
          <td className={styles.colNum}>₪{fmt(total)}</td>
        </tr>
      </tfoot>
    </table>
  );
}

// Change this to control how many PDFs can be queued at once
const PDF_UPLOAD_LIMIT = 5;

export default function EstimatePanel() {
  const [files, setFiles] = useState([]);
  const [results, setResults] = useState([]);
  const [running, setRunning] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [pagesByFile, setPagesByFile] = useState({}); // filename -> pages string
  const [showMeasurements, setShowMeasurements] = useState(false);
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

  const removeFile = (name) => {
    setFiles((prev) => prev.filter((f) => f.name !== name));
    setPagesByFile((prev) => {
      const next = { ...prev };
      delete next[name];
      return next;
    });
  };

  const setPagesFor = (name, value) =>
    setPagesByFile((prev) => ({ ...prev, [name]: value }));

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
        const data = await estimatePdf(file, pagesByFile[file.name] || "", showMeasurements);
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
              <input
                type="text"
                className={styles.pagesInput}
                placeholder="pages (e.g. 1,3 — blank = all)"
                value={pagesByFile[f.name] || ""}
                onChange={(e) => setPagesFor(f.name, e.target.value)}
                disabled={running}
              />
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

      <label className={styles.checkboxRow}>
        <input
          type="checkbox"
          checked={showMeasurements}
          onChange={(e) => setShowMeasurements(e.target.checked)}
          disabled={running}
        />
        Show segment lengths on plan
      </label>

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
                  {r.annotated_pages?.length > 0 && (
                    <div className={styles.annotations}>
                      {r.annotated_pages.map((p) => {
                        const pb = r.page_breakdowns?.find((d) => d.page === p.page);
                        return (
                          <div key={p.page} className={styles.annotBlock}>
                            {pb?.line_items?.length > 0 && (
                              <div className={styles.pageSection}>
                                {r.page_breakdowns?.length > 1 && (
                                  <p className={styles.pageLabel}>Page {p.page}</p>
                                )}
                                <BreakdownTable
                                  items={pb.line_items}
                                  totalLabel={r.page_breakdowns?.length > 1 ? "Subtotal" : "Grand Total"}
                                  total={pb.subtotal}
                                  styles={styles}
                                />
                              </div>
                            )}
                            {(() => {
                              const pageTasks = new Set(pb?.line_items?.map((i) => i.task) ?? []);
                              const pageLegend = r.legend?.filter((e) => pageTasks.has(e.task)) ?? [];
                              return pageLegend.length > 0 && (
                                <div className={styles.legend}>
                                  {pageLegend.map((entry) => (
                                    <span key={entry.task} className={styles.legendItem}>
                                      <span
                                        className={styles.legendSwatch}
                                        style={{ background: entry.color }}
                                      />
                                      {entry.task.replace(/ \(per meter\)$/i, "")}
                                    </span>
                                  ))}
                                </div>
                              );
                            })()}
                            <figure className={styles.annotPage}>
                              <img
                                className={styles.annotImg}
                                src={`data:image/png;base64,${p.image_b64}`}
                                alt={`Page ${p.page} with marked tasks`}
                              />
                              <figcaption className={styles.annotCaption}>
                                Page {p.page}
                              </figcaption>
                            </figure>
                          </div>
                        );
                      })}
                      {r.page_breakdowns?.length > 1 && r.line_items?.length > 0 && (
                        <div className={styles.pageSection}>
                          <p className={styles.pageLabel}>Summary</p>
                          <BreakdownTable
                            items={r.line_items}
                            totalLabel="Grand Total"
                            total={r.grand_total}
                            styles={styles}
                          />
                        </div>
                      )}
                    </div>
                  )}

                  {!r.annotated_pages?.length && r.line_items?.length > 0 && (
                    <div className={styles.breakdown}>
                      <BreakdownTable
                        items={r.line_items}
                        totalLabel="Grand Total"
                        total={r.grand_total}
                        styles={styles}
                      />
                    </div>
                  )}

                  <details className={styles.details}>
                    <summary className={styles.summary}>Agent reasoning</summary>
                    <pre className={styles.pre}>{r.agent_output}</pre>
                  </details>
                </>
              )}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
