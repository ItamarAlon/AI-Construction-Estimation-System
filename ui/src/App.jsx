import { useState } from "react";
import TaskPanel from "./components/TaskPanel";
import EstimatePanel from "./components/EstimatePanel";
import "./App.css";

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Construction Estimation System</h1>
      </header>
      <main className="app-main">
        <div className={`sidebar ${sidebarOpen ? "" : "sidebar--collapsed"}`}>
          <TaskPanel />
        </div>
        <button
          className="sidebarToggle"
          onClick={() => setSidebarOpen((o) => !o)}
          title={sidebarOpen ? "Hide task list" : "Show task list"}
        >
          {sidebarOpen ? "‹" : "›"}
        </button>
        <EstimatePanel />
      </main>
    </div>
  );
}
