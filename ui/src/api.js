const BASE_URL = "http://localhost:8000";

export async function fetchTasks() {
  const res = await fetch(`${BASE_URL}/tasks`);
  if (!res.ok) throw new Error("Failed to fetch tasks");
  return res.json();
}

export async function addTask(name, price, perMeter) {
  const res = await fetch(`${BASE_URL}/tasks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, price, per_meter: perMeter }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Failed to add task");
  }
  return res.json();
}

export async function deleteTask(taskName) {
  const res = await fetch(`${BASE_URL}/tasks/${encodeURIComponent(taskName)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Failed to delete task");
  }
  return res.json();
}

export async function updateTaskPrice(taskName, newPrice) {
  const res = await fetch(`${BASE_URL}/tasks/${encodeURIComponent(taskName)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ price: newPrice }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Failed to update task price");
  }
  return res.json();
}

export async function estimatePdf(file, pages = "") {
  const formData = new FormData();
  formData.append("file", file);
  if (pages && pages.trim()) {
    formData.append("pages", pages.trim());
  }
  const res = await fetch(`${BASE_URL}/estimate/upload`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Estimation failed");
  }
  return res.json();
}
