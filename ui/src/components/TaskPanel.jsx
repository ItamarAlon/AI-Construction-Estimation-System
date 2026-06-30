import { useState, useEffect } from "react";
import { fetchTasks, addTask, deleteTask, updateTaskPrice, toggleTaskType } from "../api";
import styles from "./TaskPanel.module.css";

export default function TaskPanel() {
  const [tasks, setTasks] = useState({});
  const [name, setName] = useState("");
  const [price, setPrice] = useState("");
  const [perMeter, setPerMeter] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [editingTask, setEditingTask] = useState(null);
  const [editingPrice, setEditingPrice] = useState("");

  const load = () =>
    fetchTasks()
      .then(setTasks)
      .catch(() => setError("Could not load tasks."))
      .finally(() => setLoading(false));

  useEffect(() => { load(); }, []);

  const handleAdd = async (e) => {
    e.preventDefault();
    setError("");
    if (!name.trim() || !price) return;
    try {
      await addTask(name.trim(), parseFloat(price), perMeter);
      setName("");
      setPrice("");
      setPerMeter(false);
      load();
    } catch (err) {
      setError(err.message);
    }
  };

  const handleDelete = async (taskName) => {
    setError("");
    try {
      await deleteTask(taskName);
      if (editingTask === taskName) setEditingTask(null);
      load();
    } catch (err) {
      setError(err.message);
    }
  };

  const handleToggleType = async (taskName) => {
    if (editingTask === taskName) return;
    setError("");
    try {
      await toggleTaskType(taskName);
      load();
    } catch (err) {
      setError(err.message);
    }
  };

  const startEdit = (taskName, currentPrice) => {
    setEditingTask(taskName);
    setEditingPrice(String(currentPrice));
    setError("");
  };

  const cancelEdit = () => {
    setEditingTask(null);
    setEditingPrice("");
  };

  const handleSaveEdit = async (taskName) => {
    setError("");
    try {
      await updateTaskPrice(taskName, parseFloat(editingPrice));
      setEditingTask(null);
      setEditingPrice("");
      load();
    } catch (err) {
      setError(err.message);
    }
  };

  const entries = Object.entries(tasks);

  return (
    <aside className={styles.panel}>
      <h2 className={styles.title}>Construction Tasks</h2>

      {loading ? (
        <p className={styles.hint}>Loading…</p>
      ) : entries.length === 0 ? (
        <p className={styles.hint}>No tasks yet.</p>
      ) : (
        <ul className={styles.list}>
          {entries.map(([taskName, taskPrice]) => {
            const isPerMeter = taskName.endsWith("(per meter)");
            const isEditing = editingTask === taskName;
            return (
              <li key={taskName} className={styles.item}>
                <div className={styles.itemInfo}>
                  <span className={styles.itemName}>{taskName.replace(/ \(per meter\)$/i, "")}</span>
                  {isEditing ? (
                    <div className={styles.editRow}>
                      <span className={styles.currencySymbol}>₪</span>
                      <input
                        className={styles.editInput}
                        type="number"
                        min="0"
                        step="0.01"
                        value={editingPrice}
                        onChange={(e) => setEditingPrice(e.target.value)}
                        autoFocus
                        onKeyDown={(e) => {
                          if (e.key === "Enter") handleSaveEdit(taskName);
                          if (e.key === "Escape") cancelEdit();
                        }}
                      />
                      <button
                        className={styles.saveBtn}
                        onClick={() => handleSaveEdit(taskName)}
                        title="Save"
                      >
                        ✓
                      </button>
                      <button
                        className={styles.cancelBtn}
                        onClick={cancelEdit}
                        title="Cancel"
                      >
                        ✕
                      </button>
                    </div>
                  ) : (
                    <span
                      className={styles.itemPrice}
                      onClick={() => startEdit(taskName, taskPrice)}
                      title="Click to edit price"
                    >
                      ₪{taskPrice}
                    </span>
                  )}
                  <span
                    className={isPerMeter ? styles.badgeMeter : styles.badgeUnit}
                    onClick={() => handleToggleType(taskName)}
                    title="Click to toggle per meter / per unit"
                  >
                    {isPerMeter ? "per meter" : "per unit"}
                  </span>
                </div>
                <div className={styles.itemActions}>
                  <button
                    className={styles.deleteBtn}
                    onClick={() => handleDelete(taskName)}
                    title="Remove task"
                  >
                    ✕
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      )}

      <form className={styles.form} onSubmit={handleAdd}>
        <h3 className={styles.formTitle}>Add Task</h3>
        {error && <p className={styles.error}>{error}</p>}
        <input
          className={styles.input}
          placeholder="Task name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          required
        />
        <input
          className={styles.input}
          placeholder="Price (₪)"
          type="number"
          min="0"
          step="0.01"
          value={price}
          onChange={(e) => setPrice(e.target.value)}
          required
        />
        <label className={styles.toggle}>
          <input
            type="checkbox"
            checked={perMeter}
            onChange={(e) => setPerMeter(e.target.checked)}
          />
          Per meter
        </label>
        <button className={styles.addBtn} type="submit">Add Task</button>
      </form>
    </aside>
  );
}
