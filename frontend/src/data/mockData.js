export const mockTransactions = [
  {
    id: "TXN-8923-441A",
    amount: 2450.00,
    merchant: "Luxury Electronics Direct",
    location: "Miami, FL (IP mismatch)",
    time: "02:45 AM",
    userHistory: "Typically spends <$50/week",
    category: "Electronics",
    date: "Today",
    riskScore: 88, // 0-100
    riskLevel: "severe", // safe, suspicious, severe
    reasons: [
      "Transaction amount is unusually high",
      "Occurred at an uncommon time (Late Night)",
      "Location mismatch with primary IP address",
      "Pattern is rare for this type of user"
    ],
    financialImpact: 2450.00,
    confidence: "Very High",
    uncertainty: "Low",
    status: "pending" // pending, approved, blocked, reviewing
  },
  {
    id: "TXN-1120-998B",
    amount: 12.50,
    merchant: "Starbucks Store #441",
    location: "Seattle, WA",
    time: "08:15 AM",
    userHistory: "Daily customer",
    category: "Food & Beverage",
    date: "Today",
    riskScore: 5,
    riskLevel: "safe",
    reasons: [
      "Amount matches historical patterns",
      "Merchant is frequently visited by user",
      "Location and time align with expected behavior"
    ],
    financialImpact: 0,
    confidence: "High",
    uncertainty: "Very Low",
    status: "approved"
  },
  {
    id: "TXN-5531-229C",
    amount: 320.00,
    merchant: "Steam Games Online",
    location: "Unknown / Proxy",
    time: "11:30 PM",
    userHistory: "Occasional gamer, usually <$60",
    category: "Digital Goods",
    date: "Yesterday",
    riskScore: 65,
    riskLevel: "suspicious",
    reasons: [
      "Transaction amount is higher than usual for this category",
      "Using an anonymizing Proxy/VPN",
      "Velocity: 3rd transaction today"
    ],
    financialImpact: 320.00,
    confidence: "Medium",
    uncertainty: "Moderate",
    status: "reviewing"
  }
];

export const edaInsights = [
  "Fraud is 4x more frequent between Midnight and 4 AM.",
  "Transactions over $1,500 have a 60% higher risk profile.",
  "Digital Goods & Electronics show the most anomalous activity."
];

export const modelAgreement = {
  agreed: true,
  summary: "Most checks agree this is risky",
  details: [
    { name: "Check A (Tree)", status: "flagged" },
    { name: "Check B (Distance)", status: "flagged" },
    { name: "Check C (Rules)", status: "passed" }
  ]
};
