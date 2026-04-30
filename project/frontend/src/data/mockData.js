export const pipelineRuns = [
  {
    id: "RUN-9821",
    model: "XGBoost Fraud Classifier",
    dataset: "q3_transactions_v2.csv",
    status: "success",
    accuracy: 94.2,
    f1Score: 0.89,
    latency: "42ms",
    time: "2 mins ago",
    author: "Auto-trigger (PR #421)",
    stages: [
      { name: "Ingestion", status: "done" },
      { name: "EDA & Prep", status: "done" },
      { name: "Training", status: "done" },
      { name: "Validation", status: "done" },
      { name: "Deploy", status: "done" }
    ]
  },
  {
    id: "RUN-9820",
    model: "Graph Neural Net (GNN)",
    dataset: "user_relations_full.csv",
    status: "failed",
    accuracy: null,
    f1Score: null,
    latency: null,
    time: "45 mins ago",
    author: "data_eng_bot",
    stages: [
      { name: "Ingestion", status: "done" },
      { name: "EDA & Prep", status: "failed", error: "Missing feature column: 'ip_subnet'" },
      { name: "Training", status: "pending" },
      { name: "Validation", status: "pending" },
      { name: "Deploy", status: "pending" }
    ]
  },
  {
    id: "RUN-9819",
    model: "Random Forest Baseline",
    dataset: "q3_transactions_v1.csv",
    status: "running",
    accuracy: null,
    f1Score: null,
    latency: null,
    time: "In progress",
    author: "sarah_ds",
    stages: [
      { name: "Ingestion", status: "done" },
      { name: "EDA & Prep", status: "done" },
      { name: "Training", status: "running" },
      { name: "Validation", status: "pending" },
      { name: "Deploy", status: "pending" }
    ]
  }
];

export const edaMetrics = [
  { feature: "Transaction Amount", importance: 0.35, drift: "+2.1%" },
  { feature: "Time of Day", importance: 0.22, drift: "-0.5%" },
  { feature: "IP Distance", importance: 0.18, drift: "+5.4%" },
  { feature: "Merchant Category", importance: 0.15, drift: "0.0%" },
  { feature: "Velocity (1hr)", importance: 0.10, drift: "+1.2%" }
];

export const performanceData = [
  { epoch: 10, loss: 0.65, accuracy: 70 },
  { epoch: 20, loss: 0.50, accuracy: 78 },
  { epoch: 30, loss: 0.35, accuracy: 85 },
  { epoch: 40, loss: 0.25, accuracy: 91 },
  { epoch: 50, loss: 0.20, accuracy: 94 },
];

export const driftData = [
  { day: "Mon", score: 0.02 },
  { day: "Tue", score: 0.03 },
  { day: "Wed", score: 0.02 },
  { day: "Thu", score: 0.05 },
  { day: "Fri", score: 0.08 },
  { day: "Sat", score: 0.12 },
  { day: "Sun", score: 0.15 },
];
