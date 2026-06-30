# Construction Estimation System

AI-powered construction cost estimator. Upload a PDF floor plan and get a priced breakdown of demolition, building, and finishing tasks.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- An OpenRouter API key

## Setup

1. Rename `.env.example` to `.env` and add your API key:

```
OPENROUTER_API_KEY=your_key_here
```

## Running

```bash
docker compose up --build
```

The first build takes a few minutes while Python dependencies are installed. Subsequent starts are fast.

Once running:
- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000

The frontend will not start until the backend is healthy, so both will be ready by the time the browser loads.

## Stopping

```bash
docker compose down
```

## Usage

1. Open http://localhost:3000
2. Drag and drop one or more PDF construction plans onto the upload area (up to 5 at once)
3. Optionally specify which pages to analyze (e.g. `1,3`) — leave blank to analyze all pages
4. Click **Run Estimation**
5. Results show a cost breakdown per task, annotated plan images, and the agent's reasoning

### Managing tasks and prices

Use the task panel to add, edit, or remove tasks and their unit prices. Changes take effect on the next estimation run.
