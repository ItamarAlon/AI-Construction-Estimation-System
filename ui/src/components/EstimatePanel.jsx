import { useState, useRef, useCallback, useEffect } from "react";
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
  const [hiddenTasks, setHiddenTasks] = useState(new Set());
  const [removedTasks, setRemovedTasks] = useState(new Set());
  const [showMeasurements, setShowMeasurements] = useState(false);
  const [contextMenu, setContextMenu] = useState(null); // { x, y, task }
  const inputRef = useRef(null);

  const toggleTask = useCallback((task) => {
    setHiddenTasks((prev) => {
      const next = new Set(prev);
      if (next.has(task)) next.delete(task);
      else next.add(task);
      return next;
    });
  }, []);

  const openContextMenu = useCallback((e, task) => {
    e.preventDefault();
    setContextMenu({ x: e.clientX, y: e.clientY, task });
  }, []);

  const removeTask = useCallback((task) => {
    setRemovedTasks((prev) => new Set([...prev, task]));
    setContextMenu(null);
  }, []);

  useEffect(() => {
    if (!contextMenu) return;
    const dismiss = () => setContextMenu(null);
    document.addEventListener("mousedown", dismiss);
    return () => document.removeEventListener("mousedown", dismiss);
  }, [contextMenu]);

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
        const data = await estimatePdf(file, pagesByFile[file.name] || "");
        out.push({ name: file.name, ...data, error: null });
      } catch (err) {
        out.push({ name: file.name, error: err.message });
      }
      setResults([...out]);
    }
    setRunning(false);
    setFiles([]);
    setPagesByFile({});
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
              ) : !r.line_items?.length && !r.annotated_pages?.length ? (
                <p className={styles.resultError}>
                  No construction tasks found. Make sure this is a construction plan PDF.
                </p>
              ) : (
                <>
                  {r.annotated_pages?.length > 0 && (() => {
                    const summaryItems = (r.line_items ?? []).filter((i) => !removedTasks.has(i.task));
                    const summaryTotal = summaryItems.reduce((s, i) => s + (i.cost ?? 0), 0);
                    return (
                      <div className={styles.annotations}>
                        {r.page_breakdowns?.length > 1 && summaryItems.length > 0 && (
                          <div className={styles.pageSection}>
                            <p className={styles.pageLabel}>Summary</p>
                            <BreakdownTable
                              items={summaryItems}
                              totalLabel="Grand Total"
                              total={summaryTotal}
                              styles={styles}
                            />
                          </div>
                        )}
                        {r.annotated_pages.map((p) => {
                          const pb = r.page_breakdowns?.find((d) => d.page === p.page);
                          const pageItems = (pb?.line_items ?? []).filter((i) => !removedTasks.has(i.task));
                          const pageTotal = pageItems.reduce((s, i) => s + (i.cost ?? 0), 0);
                          return (
                            <div key={p.page} className={styles.annotBlock}>
                              {pageItems.length > 0 && (
                                <div className={styles.pageSection}>
                                  {r.page_breakdowns?.length > 1 && (
                                    <p className={styles.pageLabel}>Page {p.page}</p>
                                  )}
                                  <BreakdownTable
                                    items={pageItems}
                                    totalLabel={r.page_breakdowns?.length > 1 ? "Subtotal" : "Grand Total"}
                                    total={r.page_breakdowns?.length > 1 ? pageTotal : summaryTotal}
                                    styles={styles}
                                  />
                                </div>
                              )}
                              {(() => {
                                const pageTasks = new Set(pb?.line_items?.map((i) => i.task) ?? []);
                                const pageLegend = (r.legend ?? []).filter(
                                  (e) => pageTasks.has(e.task) && !removedTasks.has(e.task)
                                );
                                const hasMeasurements = Object.keys(p.measurement_layers ?? {}).some(
                                  (t) => !removedTasks.has(t)
                                );
                                return (pageLegend.length > 0 || hasMeasurements) && (
                                  <div className={styles.legend}>
                                    {pageLegend.map((entry) => {
                                      const hidden = hiddenTasks.has(entry.task);
                                      return (
                                        <span
                                          key={entry.task}
                                          className={`${styles.legendItem} ${hidden ? styles.legendItemHidden : ""}`}
                                          onClick={() => toggleTask(entry.task)}
                                          onContextMenu={(e) => openContextMenu(e, entry.task)}
                                          title={hidden ? "Click to show" : "Click to hide"}
                                        >
                                          <span
                                            className={styles.legendSwatch}
                                            style={{ background: entry.color }}
                                          />
                                          {entry.task.replace(/ \(per meter\)$/i, "")}
                                        </span>
                                      );
                                    })}
                                    {hasMeasurements && (
                                      <span
                                        className={`${styles.legendItem} ${styles.legendItemMeasure} ${showMeasurements ? "" : styles.legendItemHidden}`}
                                        onClick={() => setShowMeasurements((v) => !v)}
                                        title={showMeasurements ? "Hide segment lengths" : "Show segment lengths"}
                                      >
                                        <span className={`${styles.legendSwatch} ${styles.measureSwatch}`} />
                                        {showMeasurements ? "hide lengths" : "show lengths"}
                                      </span>
                                    )}
                                  </div>
                                );
                              })()}
                              <figure className={styles.annotPage}>
                                <div className={styles.imageStack}>
                                  <img
                                    className={styles.annotImg}
                                    src={`data:image/png;base64,${p.base_image_b64}`}
                                    alt={`Page ${p.page}`}
                                  />
                                  {Object.entries(p.task_layers ?? {}).map(([task, b64]) =>
                                    !hiddenTasks.has(task) && !removedTasks.has(task) && (
                                      <img
                                        key={task}
                                        className={styles.overlayImg}
                                        src={`data:image/png;base64,${b64}`}
                                        alt=""
                                      />
                                    )
                                  )}
                                  {showMeasurements && Object.entries(p.measurement_layers ?? {}).map(([task, b64]) =>
                                    !hiddenTasks.has(task) && !removedTasks.has(task) && (
                                      <img
                                        key={`meas-${task}`}
                                        className={styles.measureOverlayImg}
                                        src={`data:image/png;base64,${b64}`}
                                        alt=""
                                      />
                                    )
                                  )}
                                </div>
                                <figcaption className={styles.annotCaption}>
                                  Page {p.page}
                                </figcaption>
                              </figure>
                            </div>
                          );
                        })}
                      </div>
                    );
                  })()}

                  {!r.annotated_pages?.length && r.line_items?.length > 0 && (() => {
                    const items = (r.line_items ?? []).filter((i) => !removedTasks.has(i.task));
                    const total = items.reduce((s, i) => s + (i.cost ?? 0), 0);
                    return items.length > 0 && (
                      <div className={styles.breakdown}>
                        <BreakdownTable
                          items={items}
                          totalLabel="Grand Total"
                          total={total}
                          styles={styles}
                        />
                      </div>
                    );
                  })()}

                </>
              )}
            </div>
          ))}
        </div>
      )}

      {contextMenu && (
        <div
          className={styles.contextMenu}
          style={{ top: contextMenu.y, left: contextMenu.x }}
          onMouseDown={(e) => e.stopPropagation()}
        >
          <button className={styles.contextMenuItem} onClick={() => removeTask(contextMenu.task)}>
            Remove task
          </button>
        </div>
      )}
    </section>
  );
}
