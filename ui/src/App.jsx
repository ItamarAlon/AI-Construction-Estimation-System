import TaskPanel from "./components/TaskPanel";
import EstimatePanel from "./components/EstimatePanel";
import "./App.css";

export default function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>Construction Estimation System</h1>
      </header>
      <main className="app-main">
        <TaskPanel />
        <EstimatePanel />
      </main>
    </div>
  );
}
