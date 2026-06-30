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

export async function toggleTaskType(taskName) {
  const res = await fetch(`${BASE_URL}/tasks/${encodeURIComponent(taskName)}/toggle-type`, {
    method: "PATCH",
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Failed to toggle task type");
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

export async function renameTask(taskName, newName) {
  const res = await fetch(`${BASE_URL}/tasks/${encodeURIComponent(taskName)}/rename`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ new_name: newName }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Failed to rename task");
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

export async function estimatePdf(file, pages = "", scaleFactor = 1.0, onNode = null) {
  const formData = new FormData();
  formData.append("file", file);
  if (pages && pages.trim()) formData.append("pages", pages.trim());
  if (scaleFactor !== 1.0) formData.append("scale_factor", String(scaleFactor));

  const res = await fetch(`${BASE_URL}/estimate/upload/stream`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Estimation failed");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop(); // keep last incomplete line
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const event = JSON.parse(line.slice(6));
      if (event.type === "node" && onNode) {
        onNode(event.node);
      } else if (event.type === "result") {
        return event.data;
      } else if (event.type === "error") {
        throw new Error(event.message);
      }
    }
  }
  throw new Error("Stream ended without result");
}
